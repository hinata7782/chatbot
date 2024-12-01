import torch
import torch.nn as nn
from transformers import BertModel, BertJapaneseTokenizer


class Encoder(nn.Module):
    def __init__(self, model_name='cl-tohoku/bert-base-japanese'):
        super(Encoder, self).__init__()
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)

    
    def forward(self, input_text, input_len=None):
        # テキストデータをトークナイズしてBERT入力用のテンソルに変換
        encoded_inputs = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoded_inputs['input_ids']
        attention_mask = encoded_inputs['attention_mask']

        # BERTモデルへの入力
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state     # 各トークンの隠れ層（数値の集まりを出力する層)出力
        pooler_output = outputs.pooler_output             # [CLS]トークンの隠れ層出力

        #出力を分割して格納
        hidden_states = last_hidden_state
        cell_states = pooler_output

        return hidden_states, cell_states