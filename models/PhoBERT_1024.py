import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import torch

class FakenewsClassifier(nn.Module):
    def __init__(self, n_classes, dropout_rate=0.3):
        super(FakenewsClassifier, self).__init__()
        # Khởi tạo phoBERT
        self.bert = AutoModel.from_pretrained("vinai/phobert-base")

        # Classifier network
        self.d1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 64)
        self.bn1 = nn.LayerNorm(64)
        self.d2 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(64, n_classes)
        
        # Khởi tạo trọng số theo normal
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.normal_(self.fc2.bias, 0)

    def forward(self, input_ids, attention_mask):
        last_hidden_state, output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )

        x = self.d1(output)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.d2(x)
        x = self.fc2(x)
        return x
    
class Lilos_FakenewsClassifier(nn.Module):
    def __init__(self, n_classes, dropout_rate=0.3):
        super(Lilos_FakenewsClassifier, self).__init__()
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.model = FakenewsClassifier(n_classes=self.n_classes, dropout_rate=self.dropout_rate)
        
        # Classifier
        self.classifier = nn.Linear(in_features=15, out_features=3)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, b_encoding_texts, device):
        extract_features = []
        # Duyệt qua từng sub-text trong texts
        for b_encoding_text in b_encoding_texts:
            b_encoding_text_input_ids = b_encoding_text['input_ids'].to(device)
            b_encoding_text_attention_masks = b_encoding_text['attention_mask'].to(device)
            # Sử dụng tính năng AMP để giảm tải bộ nhớ
            with torch.cuda.amp.autocast():
                # Trích xuất đặc trưng cho mỗi encoding
                features_text = self.model(b_encoding_text_input_ids, attention_mask=b_encoding_text_attention_masks)
                extract_features.append(features_text)
        
        # Gộp các đặc trưng đã trích xuất từ các sub-text lại thành một tensor duy nhất
        extract_features = torch.cat(extract_features, dim=1) 
        # Tiến hành phân loại
        outputs = self.softmax(self.classifier(extract_features.float()))
            
        # Trả về kết quả
        return outputs