import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def ingest_data():
    print("--- Đang nạp tri thức vào não bộ Chat LCB... ---")
    
    # 1. Load dữ liệu (Mã hóa UTF-8)
    pdf_loader = DirectoryLoader('./data/', glob="./*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader('./data/', glob="./*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    
    documents = pdf_loader.load() + txt_loader.load()
    
    if not documents:
        print("Lỗi: Không tìm thấy file nào trong thư mục data/.")
        return

    # 2. Cắt nhỏ dữ liệu (Chunk size 500 là chuẩn nhất cho mô hình nhỏ)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"Đã cắt dữ liệu thành {len(texts)} đoạn nhỏ.")

    # 3. Tạo Embeddings và lưu vào ChromaDB
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    persist_directory = "./chroma_db"
    
    # Xóa não bộ cũ nếu có để làm mới hoàn toàn
    import shutil
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    print("Đang tạo Vector Store mới...")
    vector_db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"--- Hoàn thành! Chat LCB đã sẵn sàng với {len(texts)} đoạn tri thức mới. ---")

if __name__ == "__main__":
    ingest_data()
