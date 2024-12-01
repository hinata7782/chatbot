import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from excel_text import find_most_similar_question, excel_data_from_questions
from encoder_decoder import Seq2Seq
import pandas as pd


def generated_text(user_message):
    # トークナイザのインスタンスを作成
    model_name = 'cl-tohoku/bert-base-japanese'  # 使用するモデル名
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # モデルのハイパーパラメータを設定
    target_vocab_size = tokenizer.vocab_size   # 適切な語彙サイズに設定してください
    n_layers = 3
    kernel_size = 3
    embedding_size = 768
    hidden_size = 768
    zoneout = 0.5
    training = True
    dropout = 0.3

    # モデルのインスタンスを作成
    model = Seq2Seq(target_vocab_size, n_layers, kernel_size, hidden_size, embedding_size, zoneout, training, dropout, model_name)

    # ユーザーからの質問のテキストを入力し、cell_states と hidden_states を取得
    input_text =  user_message
    decoder_inputs = input_text

    # 1.入力テキストのcell_states と hidden_states の出力を生成
    with torch.no_grad():
        cell_states, hidden_states = model(input_text, decoder_inputs)

    # ユーザーからの質問のステートをまとめる
    user_states = {
        "cell_states": cell_states,
        "hidden_states": hidden_states
    }

    # Excelファイルの読み込み
    file_path = 'data\\chatbot1-data.csv'
    df = pd.read_csv(file_path)
    questions = df["question"].tolist()
    answers = df["answer"].tolist()  

    # 2.自作Excelファイルの質問データからcell_states と hidden_states を取得
    output_states = excel_data_from_questions(model, questions, answers)

    # 最も類似度の高い質問を見つける
    most_similar_question, most_similar_answer, highest_similarity_score = find_most_similar_question(user_states, output_states)
        
    if highest_similarity_score < 0.5:
        reply_text = "その質問に対する答えは分からないです。ごめんなさい。"
    else:
        reply_text = most_similar_answer

    return {"reply": reply_text}