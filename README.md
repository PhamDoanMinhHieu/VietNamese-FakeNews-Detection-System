# 🚀 XÂY DỰNG HỆ THỐNG PHÁT HIỆN TIN TỨC THẬT GIẢ
Xây dựng hệ thống phát hiện các nội dung thật giả dựa vào các cái tài liệu và nguồn dữ liệu Tiếng Việt  
Hệ thống sử dụng mô hình PhoBERT phát hiện tin tức giả kèm hệ thống RAG để tìm kiếm các tài liệu liên quan  
Truy cập dữ liệu, mô hình, tài liệu tại đây: https://drive.google.com/drive/folders/1Kf7g33Kz-mHR7p07X6Y7oomN8h3V7-0p?usp=sharing
!['Hệ thống phát hiện tin tức thật giả'](images/system.jpg)

# 🔧 XÂY DỰNG NGUỒN DỮ LIỆU
+ Nguồn dữ liệu tin thật được thu thập từ: thanhnien.vn, vnexpress.net, vietnamnet.vn, ...  
+ Nguồn dữ liệu tin giả được thu thập từ: viettan.org, danlambao.org, ...  

# 🔧 TIỀN XỬ LÝ DỮ LIỆU
!['Tiền xử lý dữ liệu'](images/processing.jpg)
Tiền sử lý dữ liệu bao gồm 5 công đoạn chính:  
==> Chuyển tất cả định dạng về chữ thường  
==> Xử lý các ký tự đặc biệt  
==> Gộp các từ liên kết nghĩa với nhau  
==> Xóa các từ không mang nhiều ý nghĩa  

# 🔧 XÂY DỰNG MÔ HÌNH
!['Mô hình dự đoán'](images/model.jpg)
Sử dụng PhoBERT để phát hiện tin tức thật giả, nhưng mô hình này chỉ nhận tối đa 256 tokens  
cho một lần inference, vì vậy tôi sử dụng kỹ thuật trung bình cộng dự đoán các patch để dự đoán  

# 🔧 CÁC THƯ VIỆN CẦN QUAN TRỌNG CÀI ĐẶT
+ torch  
+ flask  
+ transformers  
+ numpy  
+ pandas  
+ sklearn  
+ nltk  

# 📁 TỔ CHỨC CẤU TRÚC THƯ MỤC DỮ LIỆU
/checkpoints: lưu trữ trọng số mô hình  
/datacsv: lưu trữ dữ liệu news và fakenews dưới dạng file csv  
/utils: chưa các hàm thực hiện các chức năng nhất định  

# ⚡ Quick Start
!['Giao diện web'](images/web.jpg)

==> python main.py  