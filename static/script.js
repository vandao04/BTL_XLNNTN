document.addEventListener('DOMContentLoaded', function () {
    // 1. Hiển thị số lượng câu tóm tắt theo giá trị thanh trượt
    const numSentencesSlider = document.getElementById('num_sentences');
    const numSentencesOutput = numSentencesSlider.nextElementSibling;
    numSentencesOutput.textContent = numSentencesSlider.value;

    numSentencesSlider.addEventListener('input', function () {
        numSentencesOutput.textContent = numSentencesSlider.value;
    });

    // 2. Xử lý sự kiện tải lên file
    const fileInput = document.getElementById('file');
    const fileLabel = document.querySelector('.custom-file-label');

    fileInput.addEventListener('change', function () {
        const fileName = fileInput.files[0] ? fileInput.files[0].name : 'Chưa chọn file';
        fileLabel.textContent = fileName;
    });

    // 3. Xử lý hiển thị kết quả tóm tắt hoặc lỗi
    const summarySection = document.querySelector('.output-section');
    if (summarySection) {
        const errorMessage = summarySection.querySelector('p');
        if (errorMessage && errorMessage.textContent.includes('Lỗi:')) {
            alert('Có lỗi xảy ra khi tóm tắt văn bản. Vui lòng thử lại!');
        }
    }
});
