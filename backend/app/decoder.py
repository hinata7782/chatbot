import torch.nn as nn
import torch


class Decoder(nn.Module):
    def __init__(self, qrnn_layer, n_layers, kernel_size,
                hidden_size, embedding_size, target_vocab_size,
                zoneout, training, dropout):

        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(target_vocab_size, embedding_size)
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        layers = []

        for layer_idx in range(n_layers):
            input_size = embedding_size if layer_idx == 0 else hidden_size
        
            use_attention = (layer_idx == n_layers - 1)

            layers.append(qrnn_layer(input_size, hidden_size, kernel_size, use_attention,
                                    zoneout, training, dropout))
        
        self.layers = nn.Sequential(*layers)

    def init_hidden(self, inputs):
        hidden = torch.zeros(self.n_layers, inputs.size(0), self.hidden_size)
        return hidden.cuda() if torch.cuda.is_available() else hidden

    def init_weight(self):
        self.embedding.weight.data = nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, decoder_inputs, init_states, memories):
        # 層数と状態、メモリの数が一致するかチェック
        assert len(self.layers) == len(init_states) == len(memories)

        embedded_inputs = self.embedding(decoder_inputs)

        cell_states = []
        hidden_states = []

        for layer_idx, layer in enumerate(self.layers):

            state = init_states[layer_idx]
            memory = memories[layer_idx]
            c, h = layer(embedded_inputs, state, memory, layer_idx)    # QRNN層に入力
            cell_states.append(c)    # 各QRNN層のcell_stateを保存
            hidden_states.append(h)   # 各QRNN層のhidden_stateを保存


        return cell_states, hidden_states