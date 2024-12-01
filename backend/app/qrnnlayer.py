import torch
import torch.nn as nn

class QRNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, use_attention=False,
                zoneout=0.5, training=True, dropout=0.5):
        super(QRNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.use_attention = use_attention

        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=3 * hidden_size,
                                kernel_size=kernel_size)

        self.conv_linear = nn.Linear(hidden_size, 3 * hidden_size)
        self.rnn_linear = nn.Linear(2 * hidden_size, hidden_size)
        self.zoneout = zoneout
        self.training = training
        self.dropout = dropout

    def _conv_step(self, inputs, memory=None, layer_idx=0):
        inputs_ = inputs.transpose(1, 2)
        padded = nn.functional.pad(inputs_.unsqueeze(2), (self.kernel_size-1, 0, 0, 0)).squeeze(2)
        gates = self.conv1d(padded).transpose(1, 2)

        if layer_idx > 0 and memory is not None:
            gates += self.conv_linear(memory).unsqueeze(1)

        Z, F, O = gates.split(split_size=self.hidden_size, dim=2)
        return torch.tanh(Z), torch.sigmoid(F), torch.sigmoid(O)

    def _rnn_step(self, z, f, o, c, attention_memory=None):
        if c is None:
            c_ = (1 - f) * z
        else:
            # cのサイズを [batch_size, sequence_length, hidden_size] に拡張
            c = c.expand(-1, f.size(1), -1)

            c_ = f * c + (1 - f) * z

        if not self.use_attention:
            return c_, (o * c_)

        alpha = nn.functional.softmax(torch.bmm(c_, attention_memory.transpose(1, 2)).squeeze(1), dim=-1)
        context = torch.sum(alpha.unsqueeze(-1) * attention_memory, dim=1)
        h_ = self.rnn_linear(torch.cat([c_.squeeze(1), context], dim=1)).unsqueeze(1)

        return c_, (o * h_)

    def forward(self, inputs, state, memory, layer_idx):
        if layer_idx == 0 and state is not None and state.numel() > 0:
            c = None
        else:
            c = state.unsqueeze(1) if state is not None else None

        if layer_idx == 0 and memory is not None and memory.numel() > 0:
            (conv_memory, attention_memory) = (None, None)
        else:
            (conv_memory, attention_memory) = (memory, memory) 


        Z, F, O = self._conv_step(inputs, conv_memory)
        if self.training:
            mask = torch.bernoulli(F.new_ones(F.size()) * (1 - self.zoneout))
            F *= mask

        c_time, h_time = [], []

        for time, (z, f, o) in enumerate(zip(Z.split(1, 1), F.split(1, 1), O.split(1, 1))):
            c, h = self._rnn_step(z, f, o, c, attention_memory)

            if self.dropout != 0 and self.training:
                c = nn.functional.dropout(c, p=self.dropout, training=self.training, inplace=False)
            c_time.append(c)
            h_time.append(h)

        return torch.cat(c_time, dim=1), torch.cat(h_time, dim=1)
