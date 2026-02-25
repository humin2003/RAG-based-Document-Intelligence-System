# Hệ thống xử lý Tài liệu dựa trên RAG (RAG-based Document Intelligence System)

Một trợ lý ảo thông minh được xây dựng để trích xuất, xử lý và trả lời câu hỏi từ các tài liệu PDF bằng kiến trúc RAG (Retrieval-Augmented Generation). 

Dự án này cho phép người dùng tải lên nhiều file PDF cùng lúc và tương tác với nội dung bên trong thông qua giao diện chat, đảm bảo câu trả lời chính xác, sát ngữ cảnh và không bịa đặt thông tin (hallucination).

## Tính năng chính
* **Xử lý nhiều file PDF:** Trích xuất văn bản từ nhiều tài liệu PDF cùng lúc.
* **Phân chia văn bản thông minh:** Sử dụng `RecursiveCharacterTextSplitter` của LangChain để chia các tài liệu lớn thành các đoạn văn bản có nghĩa, có độ gối đầu (overlap) để tối ưu hóa việc truy xuất ngữ cảnh.
* **Nhúng và lưu trữ Vector:** Tạo các vector nhúng (embeddings) chất lượng cao bằng Google Generative AI và lưu trữ cục bộ bằng FAISS để tìm kiếm độ tương đồng nhanh chóng.
* **Trí tuệ nhân tạo đàm thoại:** Sử dụng sức mạnh của các model Google Gemini Flash để đưa ra câu trả lời chính xác dựa hoàn toàn vào ngữ cảnh tài liệu được cung cấp.
* **Giao diện tương tác:** Giao diện chat thân thiện với người dùng, tương tự ChatGPT, được xây dựng bằng Streamlit.

## Công nghệ sử dụng
* **Ngôn ngữ:** Python
* **LLM & Embeddings:** Google Gemini API (gemini-1.5-flash / gemini-2.0-flash)
* **Framework:** LangChain
* **Cơ sở dữ liệu Vector:** FAISS (Facebook AI Similarity Search)
* **Frontend:** Streamlit
* **Xử lý tài liệu:** PyPDF2

## Hướng dẫn cài đặt và chạy cục bộ

**1. Clone repository về máy**
git clone [https://github.com/humin2003/RAG-based-Document-Intelligence-System.git](https://github.com/humin2003/RAG-based-Document-Intelligence-System.git)
cd RAG-based-Document-Intelligence-System

**2. Tạo môi trường ảo và cài đặt thư viện**
python -m venv venv
source venv/bin/activate  # Trên Windows sử dụng: venv\Scripts\activate
pip install -r requirements.txt

**3. Thiết lập biến môi trường**
Tạo một file .env ở thư mục gốc và thêm Google API Key của bạn vào:
GOOGLE_API_KEY="dien_api_key_cua_ban_vao_day"

**4. Chạy ứng dụng**
streamlit run pdfquery.py

## Hướng dẫn sử dụng
* Mở đường link Streamlit cục bộ hiển thị trên terminal.

* Tải lên một hoặc nhiều tài liệu PDF ở thanh công cụ bên trái.

* Bấm "Xử lý dữ liệu" để hệ thống xây dựng cơ sở dữ liệu vector.

* Bắt đầu trò chuyện và đặt câu hỏi về tài liệu của bạn!