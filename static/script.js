document.addEventListener('DOMContentLoaded', function () {
    const form = document.querySelector('form');
    
    form.addEventListener('submit', function (event) {
        const textArea = document.querySelector('textarea[name="text"]');
        
        // Kiểm tra xem người dùng đã nhập văn bản chưa
        if (!textArea.value.trim()) {
            alert('Vui lòng nhập một đoạn văn bản để tóm tắt.');
            event.preventDefault(); // Ngăn không cho gửi biểu mẫu
        } else {
            // Thông báo khi bắt đầu tóm tắt
            alert('Đang tóm tắt văn bản...');
        }
    });
});
