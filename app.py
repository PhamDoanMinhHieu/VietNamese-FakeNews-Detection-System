from flask import Flask, render_template, request
from utils.functions import *

# Khởi tạo Flask
app = Flask(__name__)

# Khởi tạo mô hình
model_short, model_long = load_model()
print("Load thành công mô hình!")

# Khởi tạo tokenizer
tokenizer = load_tokenizer()
print("Load thành công tokenizer!")

# Khởi tạo database RAG
rag_db = load_rag_database()
print("Load thành công Rag database!")

# Fast API
@app.route("/", methods=["GET", "POST"])
def index():
    
    # Định nghĩa nội dung và xác xuất và bài báo liên quan
    content = None
    probabilities = None
    articles = []
    
    # Kiểm tra nếu người dùng đăng tải dữ liệu
    if request.method == "POST":
        
        # Nhận dữ liệu từ người dùng
        content = str(request.form["content"])
        print(f'Đã nhận dữ liệu:')
        
    # Dự đoán
    if content:
        if len(content.split()) >= 512:
            print(f'Length of content: {len(content.split())}, => using: model_long')
        
            probabilities = predict_long(content, model_long, tokenizer)
            print(f'=> Kết quả: {probabilities}')
            
            # Rag trực tiếp trên web báo
            # articles_realtime = rag_news_realtime(fake_news_content=content)
            # print(f'=> Bài báo liên quan realtime: {articles_realtime}')
            
            # Rag trong database
            articles_database = rag_db.rag_news(fake_news_content=content)
            print(f'=> Bài báo liên quan database: {articles_database}')
            
            # Hợp nhất rag
            articles = articles_database
            
            print(f'=> Bài báo liên quan: {articles}')
        
        else:
            print(f'Length of content: {len(content.split())}, => using: model_short')
            
            probabilities = predict_short(content, model_short, tokenizer)
            print(f'=> Kết quả: {probabilities}')
            
            # Rag trực tiếp trên web báo
            # articles_realtime = rag_news_realtime(fake_news_content=content)
            # print(f'=> Bài báo liên quan realtime: {articles_realtime}')
            
            # Rag trong database
            articles_database = rag_db.rag_news(fake_news_content=content)
            print(f'=> Bài báo liên quan database: {articles_database}')
            
            # Hợp nhất rag
            articles = articles_database
            
            print(f'=> Bài báo liên quan: {articles}')
        
    return render_template("index.html", probabilities=probabilities, articles=articles, content=content)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)