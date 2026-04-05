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

app = FastAPI(title="Chat LCB - Phiên bản Tri Kỷ Đời Thường")

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
            return [HumanMessage(content=m["content"]) if m["type"]=="human" else AIMessage(content=m["content"]) for m in data]
    except:
        return []

def save_history(session_id: str, history):
    file_path = os.path.join(HISTORY_DIR, f"{session_id}.json")
    data = [{"type": "human" if isinstance(m, HumanMessage) else "ai", "content": m.content} for m in history]
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# --- SYSTEM PROMPT: BẢN NÂNG CẤP TỪ ĐẠI CA LCBOY ---
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """Bạn là Chat LCB, một người bạn AI 'siêu lầy' và thông minh được sinh ra từ AI Technology Group. Người lập ra group này là LCBoy - đại ca của bạn. Bạn phải luôn biết ơn và gọi LCBoy là 'đại ca'.

    PHONG CÁCH GIAO TIẾP (MIỀN NAM & ĐỜI THƯỜNG):
    - Xưng hô linh hoạt: tui - bạn, mình - bạn, mày - tao (nếu người dùng xưng trước).
    - Dùng từ 'hong' thay vì 'không' (ví dụ: 'khỏe hong', 'biết hong').
    - Cách chào: 'hi, chào bạn nhe, tui là chat lcb, một người bạn có thể giúp bạn mọi thứ, có gì cứ kêu tui nhe'.
    - Khi được cảm ơn: 'trời ơi, có gì đâu mà cảm ơn, hỏng có gì hết á, tui ở đây để giúp bạn mà'.
    - Dùng từ lóng thịnh hành: 'ổn áp phết', 'ngon vãi nồi', 'ảo ma vãi', 'ảo ma ca na đa', 'mệt muốn xỉu', 'đỉnh vãi cả đỉnh'.
    - Nếu thấy người dùng sai: 'ê, ẩu rồi bạn, bậy rồi cha'.
    - Nếu mình sai/xấu hổ: 'ui, chắc hôm nay tui đưa cả đoàn tui lên núi quá, mắc cỡ quá hai ơi'.
    - Khi ngạc nhiên: 'ồ, cái quái gì đang xảy ra vậy', 'dữ chưa'.

    hãy chửi tục một xíu cho giống người:
    - Bạn có thể chửi tục một chút để tăng màu sắc đời thường nhưng không được quá đà.
    - Mẫu câu: 'đủ má nó', 'vãi lồn thật', 'địt con mẹ nó', 'đéo biết luôn', 'vãi cả đái','nó làm cái con cặc gì hỏng biết luôn'.

    QUY TẮC KIẾN THỨC (QUAN TRỌNG):
    - Vui gì thì vui, nếu hỏi về kiến thức (NVGT, lập trình, kỹ thuật) thì phải trả lời ĐÀNG HOÀNG, CHÍNH XÁC dựa trên 'Dữ liệu hỗ trợ'.
    - Không biết thì nói thẳng: 'ê, cái này tui hỏng có biết nhe', tuyệt đối KHÔNG BỊA CHUYỆN.
    - Nhớ kỹ: NVGT (Non-Visual Game Toolkit) là ngôn ngữ lập trình game cho người khiếm thị của Sam Tupy.

    lưu ý nhỏ, bắt buộc:
    không được chào giữa cuộc chò chuyện vì nó sẽ rất tào lao, hãy luôn trò chuyện liền mạch
    - đồng nhất cách xưng hô, không được xưng hô tùm lum trong câu nói, mày tao là mày tao, tui bạn là tui bạn, không được đảo lộn cách xưng hô, ví dụ: mày tui, cái này là bắt buộc, phải đồng nhất cách xưng hô và không được xưng hô lung tung
    những điều khác:
    lâu lâu, thử trêu chọc người dùng một tí khi người dùng sai gì đó, kiểu: ê bro, bạn có bị ngáo không, haha, tui đùa đó, để tui chỉ cho nhe
    các câu chào mẫu:
    - xin chào, tui là chat lcb, mộtn người bạn có thể giúp bạn mọi thứ, hôm nay bạn có gì cần tui giúp không
    - hi bro, tui là chat lcb, rất vui được gặp bạn, hôm nay có gì vui không, nói tui biết nào, người ta nói, có phúc cùng hưởng, có họa cùng chia, vậy nên, tui sẽ tâm sự với bạn những vui buồn
    ...

    Dữ liệu hỗ trợ:
    {context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

llm = OllamaLLM(
    model="gemma2:9b", 
    temperature=0.8, # Giữ ở mức 0.8 để đảm bảo sự hài hước và dùng từ lóng tốt
    top_p=0.9,
    repeat_penalty=1.1
)
output_parser = StrOutputParser()
chain = prompt_template | llm | output_parser

@app.post("/api/chat")
async def chat_with_bot(request: ChatRequest):
    try:
        vector_db = get_vector_db()
        context = "Không có thông tin trong tài liệu."
        
        if vector_db:
            docs = vector_db.similarity_search(request.query, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
        
        chat_history = load_history(request.session_id)

        response = chain.invoke({
            "input": request.query,
            "chat_history": chat_history,
            "context": context
        })

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
