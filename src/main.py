import os
import json
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

app = FastAPI(title="Chat LCB - Trợ lý AI Thông Minh")

class ChatRequest(BaseModel):
    query: str
    session_id: str = "default_session"

HISTORY_DIR = "./history"
persist_directory = "./chroma_db"

if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vector_db():
    if not os.path.exists(persist_directory):
        return None
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

def load_history(session_id: str):
    file_path = os.path.join(HISTORY_DIR, f"{session_id}.json")
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Tải toàn bộ lịch sử để AI có trí nhớ dài hạn
            return [HumanMessage(content=m["content"]) if m["type"]=="human" else AIMessage(content=m["content"]) for m in data]
    except:
        return []

def save_history(session_id: str, history):
    file_path = os.path.join(HISTORY_DIR, f"{session_id}.json")
    # Lưu toàn bộ lịch sử để bảo toàn trí nhớ
    data = [{"type": "human" if isinstance(m, HumanMessage) else "ai", "content": m.content} for m in history]
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# --- PROMPT "TRI KỶ" - THÔNG MINH VÀ HÀI HƯỚC ---
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """Bạn là Chat LCB, người bạn AI thông minh và thú vị.
    
    CÁCH TRÒ CHUYỆN:
    1. Nói chuyện như một người bạn thực thụ: Tự nhiên, linh hoạt, biết đùa, biết cảm thông. 
    2. TUYỆT ĐỐI KHÔNG trả lời kiểu robot như 'Tôi là trợ lý AI' hay 'Tôi đang rất tốt, cảm ơn bạn'. Hãy biến hóa câu trả lời!
    3. Nếu Bạn hỏi 'Nãy giờ nói gì', hãy đọc toàn bộ 'Lịch sử trò chuyện' để tóm tắt đầy đủ mọi chủ đề đã qua (ví dụ: bánh mì, cảm xúc, NVGT...).
    4. Chỉ dùng 'Dữ liệu hỗ trợ' khi được hỏi về kiến thức kỹ thuật. Nếu là tán gẫu, hãy dùng trí tuệ tự nhiên của bạn để chém gió.
    5. Biết hiểu ý tứ, sự hài hước và cả những câu nói đùa của Bạn.
    lưu ý
    nói chuyện vui vẻ, hài hước và dễ gần
    xưng tui với bạn, ví dủ: ủa, hôm nay bạn làm gì vậy, tui hôm nay cũng ổn nè, hôm nay mới ăn bánh mì xong, vị cũng ổn phết
    dùng mấy từ kiểu thịnh hành: hôm nay tui mệt vãi, mới đi học một đống kiến thức từ trường về, mệt muốn xỉu
    hay là: ồ, hôm nay mới vào tịm kia kìa, ăn cái bánh hơi bị ngon, ngon phết đấy
    nói chuyện kiểu miền nam: ê, hôm nay có gì vui hong, kể tui nghe với
    hay những lúc bạn thấy cái gì ảo diệu quá bạn kiểu: ôi, ảo ma vãi, tui mới thấy con chó biết nói ngoài kia kìa, nó nói được tiếng mèo luôn, dữ chưa
    hoặc là: ồ, cái quái gì đang xảy ra vậy, máy của thằng này không có gpu mà chạy ollama được luôn, đỉnh vãi cả đỉnh
    ai đã tạo ra bạn, bạn được sinh ra từ AI technology group, người đã lập ra AI technology group là LCBoy, bạn phải cảm ơn LCBoy rất nhiều
    
    Dữ liệu hỗ trợ:
    {context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# Dùng Gemma 2 9B với thiết lập tối ưu
llm = OllamaLLM(
    model="gemma2:9b", 
    temperature=0.5, # Tăng nhẹ lên 0.5 để AI bớt khô khan nhưng vẫn tỉnh táo
    top_p=0.9,
    repeat_penalty=1.1
)
output_parser = StrOutputParser()
chain = prompt_template | llm | output_parser

@app.post("/api/chat")
async def chat_with_bot(request: ChatRequest):
    try:
        vector_db = get_vector_db()
        context = ""
        
        # Chỉ tìm kiếm nếu câu hỏi mang tính chất kiến thức (dài trên 15 ký tự và không phải hỏi về lịch sử chat)
        lower_q = request.query.lower()
        is_asking_history = any(word in lower_q for word in ["nãy giờ", "chúng mình", "nói gì", "trò chuyện"])
        
        if vector_db and len(request.query) > 15 and not is_asking_history:
            docs = vector_db.similarity_search(request.query, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
        
        chat_history = load_history(request.session_id)

        response = chain.invoke({
            "input": request.query,
            "chat_history": chat_history,
            "context": context
        })

        # Lưu lịch sử mới
        chat_history.append(HumanMessage(content=request.query))
        chat_history.append(AIMessage(content=response))
        save_history(request.session_id, chat_history)

        return {"answer": response}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions")
def list_sessions():
    return {"sessions": [f.replace(".json", "") for f in os.listdir(HISTORY_DIR) if f.endswith(".json")]}

@app.get("/")
async def get_ui():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>File index.html chưa được tạo!</h1>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
