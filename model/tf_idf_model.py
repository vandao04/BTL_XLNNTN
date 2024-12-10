#--------------Thư viện--------------
from sklearn.feature_extraction.text import TfidfVectorizer
from pyvi import ViTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import re

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

# --------------Hàm tiền xử lý văn bản tiếng Việt--------------
def preprocess_text_vietnamese(text):
    text = re.sub(r'[^\w\s.,!?]', '', text)      # Loại bỏ ký tự đặc biệt như @, -, *
    text = re.sub(r'\s+', ' ', text).strip() # loại bỏ khoảng trắng thừa
    text = text.lower()                      # chuyển toàn bộ văn bản về chữ thường   
    text = ViTokenizer.tokenize(text)        # tách từ trong văn bản tiếng Việt bằng thư viện ViTokenizer
    words = text.split()                     # tách văn bản thành từng từ
    
    # Loại bỏ stopwords
    filtered_words = [word for word in words if word not in vietnamese_stopwords and word.isalnum()]
    # Ghép các từ thành đoạn văn hoàn chỉnh
    return ' '.join(filtered_words)

#--------------Hàm tóm tắt văn bản tiếng Việt--------------
def summarize_text_vietnamese(text, max_sentences=3):
    # Tách văn bản thành các câu dựa trên dấu chấm câu
    sentences = sent_tokenize(text)
    
    # Nếu số câu nhỏ hơn hoặc bằng 'max_sentences', trả về toàn bộ văn bản
    if len(sentences) <= max_sentences:
        return text

    # Tiền xử lý từng câu
    preprocessed_sentences = [preprocess_text_vietnamese(sentence) for sentence in sentences]

    # Sử dụng TF-IDF vectorizer với n-grams từ 1 đến 2
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=0.01, max_df=0.85)
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)

    # Tính điểm số cho từng câu
    sentence_scores = tfidf_matrix.sum(axis=1).A1

    # Sắp xếp thứ tự câu theo điểm số
    ranked_indices = sentence_scores.argsort()[::-1]

    # Chọn tối đa `max_sentences` câu đầu tiên
    selected_sentences = [sentences[idx] for idx in ranked_indices[:max_sentences]]

    # Giữ nguyên thứ tự xuất hiện ban đầu của các câu đã chọn
    selected_sentences.sort(key=lambda x: sentences.index(x))

    # Ghép các câu đã chọn thành bản tóm tắt
    return '. '.join(selected_sentences)

#--------------Hàm tiền xử lý văn bản tiếng Anh--------------
def preprocess_text_english(text):
    text = re.sub(r'[^\w\s.,!?]', '', text)      
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in english_stopwords and word.isalnum()]
    return ' '.join(filtered_words)

#--------------Hàm tóm tắt văn bản tiếng Anh--------------
def summarize_text_english(text, max_sentences=3):
    
    sentences = sent_tokenize(text)
    if len(sentences) <= max_sentences:
        return text

    
    preprocessed_sentences = [preprocess_text_english(sentence) for sentence in sentences]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=0.01, max_df=0.85)
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)

    sentence_scores = tfidf_matrix.sum(axis=1).A1
    ranked_indices = sentence_scores.argsort()[::-1]

    selected_sentences = [sentences[idx] for idx in ranked_indices[:max_sentences]]
    selected_sentences.sort(key=lambda x: sentences.index(x))

    return '. '.join(selected_sentences)