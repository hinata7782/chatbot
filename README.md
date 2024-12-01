# チャットボット
    ～概要～
        スマホゲームである、「ディズニー　ツイステッドワンダーランド」略称「ツイステ」のゲーム内容やキャラについて質問できるチャットボットです。

# ファイル構成
    チャットボット
        |ーfrontend
                |ーnode_modules
                |ーpublic
                |ーsrc
                    ｜ーcss
                    ｜  |ーHome.css　　　
                    ｜  |ーPage1.css
                    ｜  |ーPage2.css
                    ｜
                    ｜ーpages
                    ｜  |ーimages
                    ｜  |　　｜ー22996495.png
                    ｜  |
                    ｜  |ーHome.jsx                ホームページ
                    ｜  |ーpage1-InputField.js     メッセージの入力部分
                    ｜  |ーpage1-MessageList.js    メッセージ表示部分
                    ｜  |ーpage1.jsx               ページ1（チャット部分）
                    ｜  |ーpage2.jsx               ページ2  (質問例)
                    ｜
                    ｜ーApp.css
                    ｜ーApp.js
                    ｜ーIndex.css
                    ｜－Index.css
                    ｜－略・・・
        |ーbackend
                |ー.venv
                |ーapp
                    ｜ー_pycache_
                    ｜ーdata
                    ｜  |ーchatbot1-data.csv       excelで作成した自作データ
                    ｜  
                    ｜ーdecoder.py                 デコーダモデル
                    ｜ーencoder_decoder.py　　　　　エンコーダデコーダモデル
                    ｜ーencoder.py                 エンコーダモデル
                    ｜ーexcel_test.py              自作データの回答取得や類似度計算
                    ｜ーgenerated_text.py          message.pyから取得した文章と自作データの質問列の文章との類似度を出力し、最も高い質問と対になる回答を出力
                    ｜ーmessage.py                 page1.jsxと通信・受信をする
                    ｜ーqrnnlayer.py               デコーダモデルのQRNNレイヤー


# システム構成
![システム構成](https://github.com/user-attachments/assets/a5518c83-56cf-4a4d-b996-540691d57877)

# デモ
PowerPointでけっこう強引に作成したので、バグって見えます。気にしないでもらえるとありがたいです。
![chatbot](https://github.com/user-attachments/assets/5e9a4548-dc43-4eac-b59b-cf050e4939fa)
