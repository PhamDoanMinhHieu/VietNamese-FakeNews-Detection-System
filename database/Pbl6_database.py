import requests
from bs4 import BeautifulSoup
import mysql.connector
from rank_bm25 import BM25Okapi

class Pbl6_database():
    def __init__(self, host="localhost", user="root", password="", database="pbl6"):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.links = []
        self.news = []

        self.connection = mysql.connector.connect(
            host=self.host, # Thường là localhost
            user=self.user, # Người dùng MySQL, mặc định là root
            password=self.password, # Mật khẩu MySQL, để trống nếu không có
            database=self.database  # Tên cơ sở dữ liệu
        )
        
        # Tạo con trỏ để thực hiện truy vấn
        self.cursor = self.connection.cursor()
        
        # Thực hiện truy vấn
        self.cursor.execute("SELECT Title, Source FROM tbl_news")
        
        # Lấy tất cả kết quả từ truy vấn
        rows = self.cursor.fetchall()
        
        # Xử lý kết quả
        for row in rows:
            content, source = row
            self.news.append(content)
            self.links.append(source)
        
        # Xử lí dữ liệu
        tokenized_real_news = [doc.split(" ") for doc in self.news] 
        
        # Khởi tạo mô hình BM25
        self.bm25 = BM25Okapi(tokenized_real_news)   
        
    def rag_news(self, fake_news_content):
        # Xử lí dữ liệu
        tokenized_fake_news = fake_news_content.split(" ")
        
        # Tìm kiếm các tin tức thật
        scores = self.bm25.get_scores(tokenized_fake_news)
        
        # Lấy các chỉ số của các tin tức thật có điểm số cao nhất
        top_n = 5  # Số lượng tin tức thật bạn muốn hiển thị
        top_indices = scores.argsort()[-top_n:][::-1]  # Lấy chỉ số của top N tin tức thật
        
        # Trả về kết quả links và news sau khi được lựa chọn tương ứng
        articles = []
        for index in top_indices:
            articles.append({"title": self.news[index], "link": self.links[index]})
        
        return articles