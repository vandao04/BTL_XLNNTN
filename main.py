from flask import Flask, request, render_template
from model import tf_idf_model, bert_model
from docx import Document
import PyPDF2


# Tạo Flask app
app = Flask(__name__)

# Hàm đọc nội dung từ file TXT
def read_txt_file(file):
    return file.read().decode('utf-8')

# Hàm đọc nội dung từ file Word
def read_word_file(file):
    doc = Document(file)
    return '\n'.join([para.text for para in doc.paragraphs])

# Hàm đọc nội dung từ file PDF
def read_pdf_file(file):
    text = ""
    reader = PyPDF2.PdfReader(file)  # Không cần mở lại file
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Trang tóm tắt văn bản
@app.route('/summarize', methods=['POST'])
def summarize():
    # Lấy nội dung từ form
    text = request.form.get('text', '').strip()
    num_sentences = int(request.form.get('num_sentences', 3))
    language = request.form.get('language', 'vietnamese')
    model_choice = request.form.get('model', 'bert')
    
    uploaded_file = request.files.get('file')
    
    # Ưu tiên nội dung từ file nếu không có văn bản nhập
    if not text and uploaded_file and uploaded_file.filename != '':
        if uploaded_file.filename.endswith('.docx'):
            text = read_word_file(uploaded_file)
        elif uploaded_file.filename.endswith('.pdf'):
            text = read_pdf_file(uploaded_file)
        elif uploaded_file.filename.endswith('.txt'):
            text = read_txt_file(uploaded_file)
        else:
            return render_template('index.html', error="Chỉ chấp nhận tệp .docx, .pdf, hoặc .txt", 
                                   text=text, num_sentences=num_sentences, language=language, model=model_choice)
    
    # Nếu vẫn không có nội dung
    if not text:
        return render_template('index.html', error="Vui lòng nhập văn bản hoặc tải lên tệp để tóm tắt.",
                               num_sentences=num_sentences, language=language, model=model_choice)
    
    # Kiểm tra độ dài văn bản
    if len(text) < 100:
        return render_template('index.html', error="Văn bản quá ngắn để tóm tắt, vui lòng nhập đoạn dài hơn.",
                               text=text, num_sentences=num_sentences, language=language, model=model_choice)
    
    # Tóm tắt dựa trên lựa chọn
    try:
        if language == 'vietnamese':
            if model_choice == 'tf_idf':
                summary = tf_idf_model.summarize_text_vietnamese(text, num_sentences)
            else:
                summary = bert_model.summarize_text_bert(text, num_sentences)
        else:
            if model_choice == 'tf_idf':
                summary = tf_idf_model.summarize_text_english(text, num_sentences)
            else:
                summary = bert_model.summarize_text_bert(text, num_sentences)
    except Exception as e:
        return render_template('index.html', error=f"Đã xảy ra lỗi khi tóm tắt: {str(e)}",
                               text=text, num_sentences=num_sentences, language=language, model=model_choice)

    return render_template('index.html', summary=summary, text=text, num_sentences=num_sentences, 
                           language=language, model=model_choice)

# Trang chủ
@app.route('/')
def home():
    return render_template('index.html')

# Chạy ứng dụng Flask
if __name__ == '__main__':
    app.run(debug=True)
