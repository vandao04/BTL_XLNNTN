#--------------Thư viện--------------
from nltk.tokenize import  sent_tokenize
import re
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity




#--------------Hàm tính toán vector BERT cho mỗi câu--------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
def preprocess_text(text):
    text = re.sub(r'[^\w\s.,!?]', '', text, flags=re.UNICODE)  # Loại bỏ ký tự đặc biệt như @, -,*
    text = re.sub(r'\s+', ' ', text).strip() # loại bỏ khoảng trắng thừa
    words = text.lower()                      # chuyển toàn bộ văn bản về chữ thường   

    # Ghép các từ thành đoạn văn hoàn chỉnh
    return words



#--------------Hàm tóm tắt văn bản tiếng Việt sử dụng BERT--------------
def summarize_text_bert(text, max_sentences=3):
    # Tách văn bản thành các câu
    sentences = sent_tokenize(text)

    #Nếu số câu nhỏ hơn hoặc bằng 'max_sentences', trả về toàn bộ văn bản
    if len(sentences) <= max_sentences:
        return sentences
        
    
    # Tiền xử lý từng câu
    sentences = [preprocess_text(sentence) for sentence in sentences]

    # Lấy embeddings cho từng câu sử dụng BERT
    embeddings = get_bert_embeddings(sentences)

    # Tính độ dài Euclidean (norm) của mỗi câu để đánh giá độ quan trọng
    similarity_matrix = cosine_similarity(embeddings)
    norm_scores = np.linalg.norm(embeddings, axis=1)
    sentence_scores = norm_scores + similarity_matrix.sum(axis=1)

    # Sắp xếp các câu theo độ dài norm giảm dần và chọn num_sentences câu quan trọng nhất
    ranked_indices = sentence_scores.argsort()[::-1][:max_sentences].astype(int)

    # Lấy ra các câu có điểm số cao nhất
    ranked_sentences = [sentences[i] for i in ranked_indices]
    
    # Giữ nguyên thứ tự xuất hiện ban đầu của các câu đã chọn
    ranked_sentences.sort(key=lambda x: sentences.index(x))

    # Ghép các câu lại thành một văn bản
    
    return " ".join(ranked_sentences)
            
        


# #--------------Hàm tóm tắt văn bản tiếng Anh sử dụng BERT--------------
# def summarize_text_english_bert(text, max_sentences=3):

#     sentences = sent_tokenize(text)
#     if len(sentences) <= max_sentences:
#         return sentences
        
    
#     sentences = [preprocess_text(sentence) for sentence in sentences]

#     embeddings = get_bert_embeddings(sentences)

#     similarity_matrix = cosine_similarity(embeddings)
#     norm_scores = np.linalg.norm(embeddings, axis=1)
#     sentence_scores = norm_scores + similarity_matrix.sum(axis=1)
    
#     ranked_indices = sentence_scores.argsort()[::-1][:max_sentences].astype(int)
#     ranked_sentences = [sentences[i] for i in ranked_indices]
    
#     return ". ".join(ranked_sentences)


