import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder 
from qrnnlayer import QRNNLayer
from transformers import BertJapaneseTokenizer

class Seq2Seq(nn.Module):
    def __init__(self, target_vocab_size, n_layers, kernel_size, hidden_size, embedding_size, zoneout, training, dropout, model_name='cl-tohoku/bert-base-japanese'):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(model_name)  # エンコーダーのインスタンスを作成
        self.decoder = Decoder(QRNNLayer, n_layers, kernel_size, hidden_size, embedding_size, target_vocab_size, zoneout, training, dropout)  # デコーダーのインスタンスを作成

        # トークナイザーの初期化
        self.tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')


    def forward(self, input_text, decoder_inputs, init_states=None, memories=None):
        # エンコーダーを通して入力テキストを処理
        hidden_states, cell_states = self.encoder(input_text)

        init_states = [cell_states] * len(self.decoder.layers) if init_states is None else init_states
        memories = [hidden_states] * len(self.decoder.layers) if memories is None else memories

        # デコーダ入力をトークナイズ（整数インデックス化）
        decoder_inputs = self.tokenizer(decoder_inputs, return_tensors="pt")["input_ids"]

        # デコーダーを通して生成
        cell_states, hidden_states = self.decoder(decoder_inputs, init_states, memories)

        return cell_states, hidden_states