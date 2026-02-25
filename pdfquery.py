import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import asyncio

# Load biến môi trường
load_dotenv()

# Cấu hình UI Streamlit (Nên đặt ở ngay đầu hàm hoặc đầu file)
st.set_page_config(page_title="PDF Chatbot")

# --- CÁC HÀM XỬ LÝ LÕI ---

@st.cache_resource # Giúp cache lại model, không phải load lại mỗi lần user hỏi
def get_embeddings():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

def get_pdf_content(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

def get_content_chunk(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    embeddings = get_embeddings()
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("FAISS_index")

def get_conversation_chain():
    prompt_template = """
    Hãy trả lời câu hỏi một cách chi tiết và chính xác dựa trên thông tin được cung cấp trong tài liệu.
    Nếu không tìm thấy câu trả lời trong tài liệu, hãy trả lời: "Không tìm thấy thông tin."
    Tuyệt đối không tự suy đoán hoặc cung cấp thông tin không có trong tài liệu.
    
    Tài liệu: \n {context} \n
    Câu hỏi: \n {question} \n
    Trả lời:
    """
    # Đồng bộ dùng gemini-2.0-flash giống main.py của bạn cho ổn định
    model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def generate_answer(user_question):
    embeddings = get_embeddings()
    # Load lại DB từ local
    new_db = FAISS.load_local("FAISS_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=10)
    
    chain = get_conversation_chain()
    # Dùng .invoke theo chuẩn mới của LangChain
    response = chain.invoke({"input_documents": docs, "question": user_question})
    return response["output_text"]

# --- GIAO DIỆN STREAMLIT ---

def main():
    st.header("Chatbot hỏi đáp dữ liệu PDF")

    # Khởi tạo session state để lưu lịch sử chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar: Xử lý Upload PDF
    with st.sidebar:
        st.title("Tải lên tài liệu:")
        pdf_docs = st.file_uploader("Chọn file PDF:", accept_multiple_files=True, type=["pdf"])
        if st.button("Xử lý dữ liệu"):
            if pdf_docs:
                with st.spinner("Đang đọc và phân tích PDF..."):
                    content = get_pdf_content(pdf_docs)
                    chunks = get_content_chunk(content)
                    get_vector_store(chunks)
                    st.success("Hoàn thành! Bạn có thể bắt đầu hỏi.")
            else:
                st.warning("Vui lòng tải lên file PDF trước khi bấm Xử lý.")

    # Hiển thị lịch sử chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Xử lý khi người dùng nhập câu hỏi
    if prompt := st.chat_input("Nhập câu hỏi của bạn về tài liệu..."):
        
        # Hiển thị câu hỏi của user
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Trợ lý ảo sinh câu trả lời
        with st.chat_message("assistant"):
            with st.spinner("Đang tìm kiếm thông tin..."):
                try:
                    reply = generate_answer(prompt)
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.error(f"Lỗi: Hãy chắc chắn bạn đã tải lên và Xử lý dữ liệu ở Sidebar trước nhé! (Chi tiết lỗi: {e})")

if __name__ == "__main__":
    main()