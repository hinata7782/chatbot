from flask import Flask, request, jsonify
import subprocess
import requests
from generated_text import generated_text
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/message', methods=['POST'])
def receive_message():
    try:
        # フロントエンドから受信したJSONデータを取得
        data = request.get_json()
        user_message = data.get("message", "")

        # generated_text 関数で処理を実行
        processed_data = generated_text(user_message)
        reply_text = processed_data.get("reply", "エラー: 返信がありません。")
        
        # フロントエンドに返信を返す
        return jsonify({"response": reply_text})
    except Exception as e:
        return jsonify({"response": f"エラーが発生しました: {str(e)}"}), 500

if __name__ == "__main__":
    app.debug = True
    app.run(host='127.0.0.1', port=5000)
