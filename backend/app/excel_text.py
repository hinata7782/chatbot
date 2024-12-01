import torch
from sklearn.metrics.pairwise import cosine_similarity
from encoder_decoder import Seq2Seq 
import numpy as np


# シーケンスを最大長にパディングする関数
def pad_sequences(tensor_list, max_token_len):
    padded_tensor_list = []
    
    for tensor in tensor_list:
        batch_size, seq_len, embedding_size = tensor.shape
        
        # トークン数が max_token_len より長い場合は切り捨て
        if seq_len > max_token_len:
            tensor = tensor[:, :max_token_len, :]
        # トークン数が max_token_len より短い場合はゼロパディング
        elif seq_len < max_token_len:
            padding_size = max_token_len - seq_len
            padding = torch.zeros(batch_size, padding_size, embedding_size, dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat((tensor, padding), dim=1)  # パディングを追加
        
        padded_tensor_list.append(tensor)
    
    # リスト内のテンソルをスタックして返す
    return torch.stack(padded_tensor_list)

# 自作のエクセルファイル「質問」から cell_states と hidden_states を取得する関数
def excel_data_from_questions(model, questions, answers):
    model.eval()
    output_states = []
    
    with torch.no_grad():
        for question, answer in zip(questions, answers):
            input_text = question
            decoder_inputs = question
            cell_states, hidden_states = model(input_text, decoder_inputs)

            # 結果をリストに保存
            output_states.append({
                "question": question,
                "answer": answer,
                "cell_states": cell_states,
                "hidden_states": hidden_states
            })
    
    return output_states

# 類似度の計算と最も類似する質問の取得
def find_most_similar_question(user_states, output_states):
    highest_similarity_score = -1
    most_similar_question = None
    most_similar_answer = None

    for data in output_states:
        user_cell_states = user_states["cell_states"]
        user_hidden_states = user_states["hidden_states"]
        excel_cell_states = data["cell_states"]
        excel_hidden_states = data["hidden_states"]

        # user_cell_states と excel_cell_states の第3次元（トークン数）を取得
        user_token_len = max(tensor.shape[1] for tensor in user_cell_states)
        excel_token_len = max(tensor.shape[1] for tensor in excel_cell_states)

        # 最大のトークン長を取得
        max_token_len = max(user_token_len, excel_token_len)

        #cell_states_data
        padded_user_cell_states = pad_sequences(user_cell_states, max_token_len)
        num_user_cell_states = padded_user_cell_states.numpy().reshape(1, -1)
        #hidden_states_data
        padded_user_hidden_states = pad_sequences(user_hidden_states, max_token_len)
        num_user_hidden_states = padded_user_hidden_states.numpy().reshape(1, -1)


        #cell_states
        padded_excel_cell_states = pad_sequences(excel_cell_states, max_token_len)
        num_excel_cell_states = padded_excel_cell_states.numpy().reshape(1, -1)
        #hidden_states
        padded_excel_hidden_states = pad_sequences(excel_hidden_states, max_token_len)
        num_excel_hidden_states = padded_excel_hidden_states.numpy().reshape(1, -1)


        # cell_states と hidden_states の類似度を個別に計算
        cell_similarity = cosine_similarity(num_user_cell_states,  num_excel_cell_states)
        hidden_similarity = cosine_similarity(num_user_hidden_states, num_excel_hidden_states)
        
        # 最終的な類似度は cell と hidden の平均
        average_similarity = (cell_similarity[0][0] + hidden_similarity[0][0]) / 2
        
        if average_similarity  > highest_similarity_score:
            highest_similarity_score = average_similarity 
            most_similar_question = data["question"]
            most_similar_answer = data["answer"]
    
    return most_similar_question, most_similar_answer, highest_similarity_score
