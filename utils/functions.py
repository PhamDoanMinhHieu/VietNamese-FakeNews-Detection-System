from models.PhoBERT_1024 import *
from database.Pbl6_database import *
from UITws_v1 import UITws_v1
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
import torch
import requests
from bs4 import BeautifulSoup
import re

# Hàm load RAG-database
def load_rag_database():
    my_database = Pbl6_database()
    return my_database

# Hàm load mô hình
def load_model():
    # Xác định đường dẫn
    model_long_path = "./checkpoints/LiLoPhoBERT_1024_v2.pth"
    model_short_path = "./checkpoints/PhoBERT_1024_v3.pth"
    
    # Xác định thiết bị tính toán
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Khởi tạo instance model
    model_long = Lilos_FakenewsClassifier(n_classes=3, dropout_rate=0.3)
    model_short = FakenewsClassifier(n_classes=3, dropout_rate=0.3)
    
    # Chuyển model vào thiết bị tính toán
    model_long = model_long.to(device)
    model_short = model_short.to(device)
    
    # Load model
    model_long.load_state_dict(torch.load(model_long_path, map_location=device))
    model_short.load_state_dict(torch.load(model_short_path, map_location=device))
    
    return model_short, model_long

# Hàm load tokenizer
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    return tokenizer

# Hàm segmentation words
def segmentation_words(text: str):
    uitws_v1 = UITws_v1('./checkpoints/base_sep.pkl')
    return uitws_v1.segment(texts=[text],
                            pre_tokenized=True,
                            batch_size=256)[0]
    
# Hàm remove stopword Phúc
def remove_stopwords_v1(text:str):
    # Load stopword
    stop_words = []
    with open('./checkpoints/vietnamese-stopwords.txt', 'r', encoding='utf-8') as f:
        stop_words = f.read().splitlines()

    words = text.split()
    filtered_words = [word.lower() for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Hàm remove stopword Thịnh
def remove_stopwords_v2(text):
    text = text.lower()  # Chuyển chữ hoa thành chữ thường
    
    # Danh sách các ký tự đặc biệt được phép giữ lại
    allowed_special_chars = r"!\"#$%&'()*+,\-./:;<=>?[\]^_`{|}~"
    
    # Biểu thức chính quy giữ lại chữ cái có dấu, chữ số, khoảng trắng và các ký tự đặc biệt cho phép
    # [\p{L}] giữ lại tất cả các ký tự chữ cái bao gồm cả có dấu
    allowed_chars = f"[^a-zA-Z0-9\s{re.escape(allowed_special_chars)}àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]"
    
    # Loại bỏ các ký tự không được cho phép
    text = re.sub(allowed_chars, '', text)
    
    with open('./checkpoints/vietnamese.txt', 'r', encoding='utf-8') as f:
        stop_words= set(f.read().strip().split('\n')) 
    words = text.split()  # Tách chuỗi thành danh sách từ
    filtered_words = [word for word in words if word not in stop_words]  # Lọc bỏ stop words
    text =' '.join(filtered_words)
    return text

# Hàm chuyển dữ liệu đầu vào thành dữ liệu mô hình có thể hiểu được
def convert_data_long(text:str, tokenizer: AutoTokenizer):
    # Xác định các thông số
    fix_len = 1024
    max_len = 256
    stride = 255
    
    # Remove stopwords
    text = remove_stopwords_v2(text)
    
    # Segmentation words
    text = segmentation_words(text)
    
    # Chứa các sub-tokenizer của content
    encoding_texts = []
    
    # Bắt đầu thực hiện các sub-token
    start = 0
    while start < fix_len:
        end = min(start + max_len, fix_len)
            
        # Lấy phân đoạn của văn bản từ `start` đến `end`
        segment = " ".join(text.split()[start:end])
        # print(f'segment: {segment}')
        
        # Tiến hành tokenizer
        encoding_text = tokenizer.encode_plus(
        segment,
        truncation=True,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors='pt',
        )
        
        # Thêm vào danh sách chứa các sub-tokenizer của content
        encoding_texts.append(encoding_text)
        
        # Di chuyển chỉ số bắt đầu dựa trên stride
        start += stride
        
    # Flatten encoding context
    for encoding_text in encoding_texts:
        encoding_text['input_ids'] = encoding_text['input_ids'].flatten()
        encoding_text['attention_mask'] = encoding_text['attention_mask'].flatten()
        
    # Trả về kết quả
    return {
        'encoding_texts': encoding_texts, # danh sách (input_ids, attention_mask)
    }
    
# Hàm chuyển dữ liệu đầu vào thành dữ liệu mô hình có thể hiểu được
def convert_data_short(text:str, tokenizer: AutoTokenizer):
    # Xác định các thông số
    fix_len = 1024
    max_len = 256
    stride = 256
    
    # Remove stopwords
    text = remove_stopwords_v2(text)
    
    # Segmentation words
    text = segmentation_words(text)
    
    # Chứa các sub-tokenizer của content
    encoding_texts = []
    
    # Bắt đầu thực hiện các sub-token
    start = 0
    while start < fix_len:
        end = min(start + max_len, fix_len)
            
        # Lấy phân đoạn của văn bản từ `start` đến `end`
        segment = " ".join(text.split()[start:end])
        # print(f'segment: {segment}')
        
        # Tiến hành tokenizer
        encoding_text = tokenizer.encode_plus(
        segment,
        truncation=True,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors='pt',
        )
        
        # Thêm vào danh sách chứa các sub-tokenizer của content
        encoding_texts.append(encoding_text)
        
        # Di chuyển chỉ số bắt đầu dựa trên stride
        start += stride
        
        # Kiểm tra điều kiện
        if start >= len(text.split()):
            break
        
    # Flatten encoding context
    for encoding_text in encoding_texts:
        encoding_text['input_ids'] = encoding_text['input_ids'].flatten()
        encoding_text['attention_mask'] = encoding_text['attention_mask'].flatten()
        
    # Trả về kết quả
    return {
        'encoding_texts': encoding_texts, # danh sách (input_ids, attention_mask)
    }

# Hàm dự đoán kết quả của phoBERT
def predict_short(text: str, model:FakenewsClassifier, tokenizer: AutoTokenizer):
    # Convert data
    encoding_texts = convert_data_short(text, tokenizer)['encoding_texts']
    
    # Xác định thiết bị tính toán
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Biến lưu trữ tổng của outputs_text
    total_outputs_text = None
    num_sub_texts = 0
    
    # Duyệt qua từng sub-text trong texts
    for encoding_text in encoding_texts:
        
        encoding_text_input_ids = encoding_text['input_ids'].to(device)
        encoding_text_attention_masks = encoding_text['attention_mask'].to(device)
        
        encoding_text_input_ids = encoding_text_input_ids.unsqueeze(0)
        encoding_text_attention_masks = encoding_text_attention_masks.unsqueeze(0)
        
        # Kiểm tra nếu input_ids toàn là padding token và attention_mask chủ yếu là 0
        if torch.sum(encoding_text_attention_masks) <= 2:
            continue  # Bỏ qua sub-text này
        
        # Không đạo hàm
        with torch.no_grad():
            # Dự đoán cho sub-text
            outputs_text = model(encoding_text_input_ids, attention_mask=encoding_text_attention_masks)
            num_sub_texts += 1
            
            # Cộng dồn kết quả đầu ra
            if total_outputs_text is None:
                total_outputs_text = outputs_text
            else:
                total_outputs_text += outputs_text
        
    # Tính trung bình cộng của outputs_text
    average_outputs_text = total_outputs_text / num_sub_texts

    # Áp dụng softmax để chuyển logits thành xác suất
    probabilities_softmax = torch.nn.functional.softmax(average_outputs_text, dim=-1)
    
    # Trả về kết quả
    # probabilities = {
    # "Thật": probabilities_softmax[0][0].item(),
    # "Giả do con người": probabilities_softmax[0][1].item(),
    # "Giả do AI": probabilities_softmax[0][2].item()
    # }
    
    probabilities = {
    "Thật": probabilities_softmax[0][0].item(),
    "Giả": 1 - probabilities_softmax[0][0].item()
    }
        
    return probabilities


# Hàm dự đoán kết quả của LiLophoBERT
def predict_long(text: str, model:Lilos_FakenewsClassifier, tokenizer: AutoTokenizer):
    # Convert data
    encoding_texts = convert_data_long(text, tokenizer)['encoding_texts']
    
    # Duyệt qua từng sub-text trong texts
    for encoding_text in encoding_texts:
        encoding_text['input_ids'] = encoding_text['input_ids'].unsqueeze(0)
        encoding_text['attention_mask'] = encoding_text['attention_mask'].unsqueeze(0)
    
    # Xác định thiết bị tính toán
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Không đạo hàm
    with torch.no_grad():
        # Dự đoán cho sub-text
        average_outputs_text = model(encoding_texts, device)
        
    # Áp dụng softmax để chuyển logits thành xác suất
    probabilities_softmax = average_outputs_text
    
    # Trả về kết quả
    # probabilities = {
    # "Thật": probabilities_softmax[0][0].item(),
    # "Giả do con người": probabilities_softmax[0][1].item(),
    # "Giả do AI": probabilities_softmax[0][2].item()
    # }
    
    probabilities = {
    "Thật": probabilities_softmax[0][0].item(),
    "Giả": 1- probabilities_softmax[0][0].item()
    }
        
    return probabilities
    
# Hàm tự động lấy links từ bài viết và nhiều chỉ mục trong mục lục
def auto_getlink(base_url: str, num_pages: int = 20):
    links = []
    
    for page in range(0, num_pages):
        # Tạo URL cho từng trang
        url = f"{base_url}-p{page}"  # Cấu trúc URL mới
        
        # Gửi yêu cầu HTTP
        response = requests.get(url)

        # Kiểm tra xem yêu cầu có thành công không
        if response.status_code == 200:
            # Phân tích cú pháp HTML
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Tìm tất cả các thẻ <a> có chứa link bài viết
            for a in soup.find_all('a', class_="count_cmt", href=True):
                # Lấy link
                full_link = a['href']
                
                # Thêm link đầy đủ vào danh sách
                if not full_link.startswith('http'):
                    full_link = "https://vnexpress.net" + full_link
                
                # Thêm vào danh sách link
                links.append(full_link)
        
        else:
            print(f"Không thể truy cập trang {url}. Mã lỗi: {response.status_code}")

    # Loại bỏ các link trùng nhau
    links = list(set(links))
    
    # Trả về danh sách
    return links

# Hàm tự động lấy dữ liệu từ 1 bài viết
def vnexpress(url: str):
    
    # Gửi yêu cầu HTTP
    response = requests.get(url)
    
    # Kiểm tra xem yêu cầu có thành công không
    if response.status_code == 200:
        # Phân tích cú pháp HTML
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Tìm tiêu đề bài báo
        # title = soup.find('h1', class_='title-detail').get_text(strip=True)
        title_element = soup.find('h1', class_='title-detail')
        title = title_element.get_text(strip=True) if title_element else ""
    
    # Trả về Title
    return str(title)

# Hàm auto craw data
def auto_gettext(url: str, num_pages: int):
    links = auto_getlink(url, num_pages)
    news = []
    for link in links:
        news.append(vnexpress(link))
    return links, news

# Hàm lựa chọn những tin thật liên quan đến tin giả
def rag_news_realtime(fake_news_content: str):
    
    # Lấy link và nội dung trực tiếp từ web vnexpress
    link = "https://vnexpress.net/thoi-su/chinh-tri"
    links, news = auto_gettext(url=link, num_pages=1)
    
    # Xử lí dữ liệu cho tin thật
    tokenized_real_news = [doc.split(" ") for doc in news]  # Token hóa bằng cách tách các từ
    
    # Khởi tạo mô hình BM25
    bm25 = BM25Okapi(tokenized_real_news)
    
    # Xử lí dữ liệu cho tin giả
    tokenized_fake_news = fake_news_content.split(" ")
    
    # Tìm kiếm các tin tức thật có liên quan đến tin giả
    scores = bm25.get_scores(tokenized_fake_news)
    
    # Lấy các chỉ số của các tin tức thật có điểm số cao nhất
    top_n = 5  # Số lượng tin tức thật bạn muốn hiển thị
    top_indices = scores.argsort()[-top_n:][::-1]  # Lấy chỉ số của top N tin tức thật
    
    # Trả về kết quả links và news sau khi được lựa chọn tương ứng
    articles = []
    for index in top_indices:
        articles.append({"title": news[index], "link": links[index]})
    
    return articles
    
