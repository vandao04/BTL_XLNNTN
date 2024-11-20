#--------------Thư viện--------------
from pyvi import ViTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import numpy as np
from transformers import BertTokenizer, BertModel
import torch

#--------------Tải bộ stopwords tiếng Anh từ nltk--------------
nltk.download('stopwords')
nltk.download('punkt')
english_stopwords = set(stopwords.words('english'))

#--------------Hàm đọc danh sách stopwords --------------
def load_stopwords(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        stopwords = set(word.strip() for word in f.read().splitlines())  
    return stopwords
#--------------Đọc file stopwords tiếng Việt--------------
vietnamese_stopwords = load_stopwords("vietnamese-stopwords-dash.txt")

#--------------Hàm tính toán vector BERT cho mỗi câu--------------
def get_bert_embeddings(sentences):
    # Tải bộ tokenizer và mô hình BERT đa ngôn ngữ đã được huấn luyện sẵn
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained('bert-base-multilingual-cased')
    
    embeddings = []
    # Tắt tính toán gradient để tiết kiệm bộ nhớ và tăng tốc quá trình tính toán
    with torch.no_grad():
        for sentence in sentences:
            # Tokenize câu, chuẩn bị dữ liệu đầu vào cho mô hình BERT
            inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
            # Truyền inputs vào mô hình BERT để lấy output
            outputs = model(**inputs)

            # Lấy embeddings trung bình của các token trong câu (sử dụng chiều dim=1 để tính trung bình theo chiều dài câu)
            # .detach() ngắt kết nối với đồ thị tính toán và chuyển về numpy array
            embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy())

    # Kết hợp các embeddings của từng câu thành một ma trận numpy, mỗi câu là một vector
    return np.vstack(embeddings)

#--------------Hàm tiền xử lý văn bản tiếng Việt--------------
def preprocess_text_vietnamese(text):
    text = re.sub(r'\s+', ' ', text).strip() # loại bỏ khoảng trắng thừa
    text = text.lower()                      # chuyển toàn bộ văn bản về chữ thường   
    text = ViTokenizer.tokenize(text)        # tách từ trong văn bản tiếng Việt bằng thư viện ViTokenizer
    words = text.split()                     # tách văn bản thành từng từ

    # Loại bỏ stopwords
    filtered_words = [word for word in words if word not in vietnamese_stopwords and word.isalnum()]
    # Ghép các từ thành đoạn văn hoàn chỉnh
    return ' '.join(filtered_words)



#--------------Hàm tóm tắt văn bản tiếng Việt sử dụng BERT--------------
def summarize_text_vietnamese_bert(text, num_sentences=3):
    # Tách văn bản thành các câu
    sentences = sent_tokenize(text)

    #Nếu số câu nhỏ hơn hoặc bằng 'max_sentences', trả về toàn bộ văn bản
    if len(sentences) <= num_sentences:
        return text
    
    # Lấy embeddings cho từng câu sử dụng BERT
    embeddings = get_bert_embeddings(sentences)

    # Tính độ dài Euclidean (norm) của mỗi câu để đánh giá độ quan trọng
    sentence_scores = np.linalg.norm(embeddings, axis=1)

    # Sắp xếp các câu theo độ dài norm giảm dần và chọn num_sentences câu quan trọng nhất
    ranked_indices = sentence_scores.argsort()[::-1][:num_sentences].astype(int)

    # Lấy ra các câu có điểm số cao nhất
    ranked_sentences = [sentences[i] for i in ranked_indices]

    # Ghép các câu lại thành một văn bản
    return ". ".join(ranked_sentences)

#--------------Hàm tiền xử lý văn bản tiếng Anh--------------
def preprocess_text_english(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()  
    words = word_tokenize(text)  
    filtered_words = [word for word in words if word not in english_stopwords and word.isalnum()]  
    return ' '.join(filtered_words)


#--------------Hàm tóm tắt văn bản tiếng Anh sử dụng BERT--------------
def summarize_text_english_bert(text, num_sentences=3):

    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text
    
    embeddings = get_bert_embeddings(sentences)
    sentence_scores = np.linalg.norm(embeddings, axis=1)
    
    ranked_indices = sentence_scores.argsort()[::-1][:num_sentences].astype(int)
    ranked_sentences = [sentences[i] for i in ranked_indices]
    
    return ". ".join(ranked_sentences)