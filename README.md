# RAG-based Document Intelligence System

An intelligent conversational agent built to extract, process, and answer questions from PDF documents using Retrieval-Augmented Generation (RAG) architecture. 

This project allows users to seamlessly upload multiple PDF files and interact with their content through a chat interface, ensuring accurate and context-aware responses without hallucination.

## Key Features
* **Multi-PDF Processing:** Extract text from multiple PDF documents simultaneously.
* **Smart Text Chunking:** Utilizes LangChain's `RecursiveCharacterTextSplitter` to divide large documents into meaningful, overlapping chunks for optimal context retrieval.
* **Vector Embeddings & Storage:** Generates high-quality vector embeddings using Google Generative AI and stores them locally using FAISS for rapid similarity search.
* **Conversational AI:** Powered by Google Gemini Flash models to deliver precise answers based strictly on the provided document context.
* **Interactive UI:** A user-friendly, ChatGPT-like chat interface built with Streamlit.

## Technology Stack
* **Language:** Python
* **LLM & Embeddings:** Google Gemini API (gemini-1.5-flash / gemini-2.0-flash)
* **Framework:** LangChain
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Frontend:** Streamlit
* **Document Processing:** PyPDF2

## How to Run Locally

**1. Clone the repository**
git clone [https://github.com/humin2003/RAG-based-Document-Intelligence-System.git](https://github.com/humin2003/RAG-based-Document-Intelligence-System.git)
cd RAG-based-Document-Intelligence-System

**2. Create a virtual environment & Install dependencies**
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

**3. Set up environment variables**
Create a .env file in the root directory and add your Google API Key:
GOOGLE_API_KEY="your_api_key_here"

**4. Run the application**
streamlit run pdfquery.py

## Usage

* Open the local Streamlit URL provided in the terminal.

* Upload one or more PDF documents using the sidebar.

* Click "Xử lý dữ liệu" (Process Data) to build the vector index.

* Start chatting with your documents!
