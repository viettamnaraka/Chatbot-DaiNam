from fastapi import FastAPI, Request, Response, BackgroundTasks
from pydantic import BaseModel
import time
import os
import requests
from dotenv import load_dotenv
from operator import itemgetter
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

app = FastAPI(title="API Chatbot Dai Nam", version="1.0")

# --- THONG TIN KET NOI FACEBOOK ---
xacminh = "xacminh" 
apifb_1 = "EAAPi7ZBNczcoBSplfwDN8lxs7jZBISazTVbY3lLZAfNl8gRBSQvHHxZCZBLuwTgqUQ2odaZCZBGa49WDwPknr7gdCbfVoo5hg1NOConI1eVoD1lhcLqpZBHROZBCo5jufAiYDOicwZB1ILY7BKXgT3EYVWe0QYhM7ieqcypqZBrqK3XKR00H6XimDfHe6e2ZBZBKv58iZAAXd1CA5bfQZDZD"

# --- KHOI TAO HE THONG RAG ---
load_dotenv()
print("Dang nap du lieu quy che vao VectorDB...")
quyche = TextLoader("quy_che.txt", encoding="utf-8")
taidulieu = quyche.load()

chiavanban = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
saukhichia = chiavanban.split_documents(taidulieu)

mohinhAI = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")

# Kiem tra xem may da luu san du lieu chua de tranh bat AI lam lai tu dau
if os.path.exists("./chroma_db"):
    print("⚡ Dang tai VectorDB tu o cung (Khong ton API)...")
    chrom = Chroma(persist_directory="./chroma_db", embedding_function=mohinhAI)
else:
    print("⏳ Dang nap du lieu vao VectorDB (Chi ton API chay lan dau tien)...")
    print(f"Tong so doan text can nap: {len(saukhichia)}")
    
    chrom = Chroma(embedding_function=mohinhAI, persist_directory="./chroma_db")
    kich_thuoc_nhom = 50
    for i in range(0, len(saukhichia), kich_thuoc_nhom):
        nhom_nho = saukhichia[i : i + kich_thuoc_nhom]
        print(f"Dang nap tu doan {i} den {i + len(nhom_nho)}...")
        chrom.add_documents(nhom_nho)
        
        if i + kich_thuoc_nhom < len(saukhichia):
            print("Doi 60 giay de tranh loi qua tai API cua Google...")
            time.sleep(60)
            
    print("✅ Da nap xong toan bo du lieu!")

timkiem = chrom.as_retriever(search_kwargs={"k": 3})
language = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)

# --- NÂNG CẤP PROMPT ĐỂ NHẬN BỘ NHỚ LỊCH SỬ ---
lenh = ChatPromptTemplate.from_messages([
    ("system", """Bạn là nhân viên tư vấn học vụ chuyên nghiệp của Đại học Đại Nam. 
Hãy phân tích kỹ câu hỏi và trả lời DỰA HOÀN TOÀN vào thông tin dưới đây. 

YÊU CẦU BẮT BUỘC:
1. BẮT BUỘC trả lời bằng Tiếng Việt CÓ DẤU, chuẩn ngữ pháp, thân thiện với sinh viên. Tuyệt đối không trả lời không dấu.
2. Tuyệt đối không sử dụng định dạng Markdown như dấu sao in đậm. Chỉ dùng văn bản thuần túy.
3. Nếu câu hỏi nhắc đến các vấn đề ngoài phạm vi đào tạo và thi cử trong tài liệu, TUYỆT ĐỐI KHÔNG tự bịa ra. 
Trong trường hợp đó, hãy trả lời: "Dữ liệu hiện tại của mình chưa có thông tin cụ thể về vấn đề này."

Thông tin: {ngu_canh}"""),
    #luu lich su chat
    MessagesPlaceholder(variable_name="lich_su_chat"),
    ("human", "{cau_hoi}")
])

def dinh_dang_tai_lieu(tai_lieu_dau_vao):
    return "\n\n".join(tl.page_content for tl in tai_lieu_dau_vao)
    
chuoi_rag = (
    RunnablePassthrough.assign(
        ngu_canh=itemgetter("cau_hoi") | timkiem | dinh_dang_tai_lieu
    )
    | lenh
    | language
    | StrOutputParser()
)

kho_luu_tru = {} 

def lay_lich_su_chat(session_id: str) -> BaseChatMessageHistory:
    """Tự động tạo hoặc gọi bộ nhớ cũ dựa trên ID Facebook của người nhắn"""
    if session_id not in kho_luu_tru:
        kho_luu_tru[session_id] = ChatMessageHistory()
    return kho_luu_tru[session_id]

# Gói chuỗi RAG vào trình quản lý bộ nhớ
chuoi_hoan_chinh = RunnableWithMessageHistory(
    chuoi_rag,
    lay_lich_su_chat,
    input_messages_key="cau_hoi",
    history_messages_key="lich_su_chat",
)

print("✅ He thong RAG (Da tich hop Memory) san sang!")

# --- CAC HAM XU LY FACEBOOK MESSENGER ---

def lam_sach_markdown(text: str) -> str:
    """Hàm lọc và chuyển đổi các ký tự markdown của AI sang định dạng đẹp trên Messenger"""
    text = text.replace("**", "")
    text = text.replace("- ", "• ")
    text = text.replace("+ ", "  - ")
    return text

def gui_tin_nhan_facebook(id_nguoi_gui: str, noi_dung_tin_nhan: str):
    """Ham gui tin nhan tra lai cho nguoi dung qua Messenger API"""
    apifb_2 = f"https://graph.facebook.com/v20.0/me/messages?access_token={apifb_1}"
    tradulieu = {
        "recipient": {"id": id_nguoi_gui},
        "message": {"text": noi_dung_tin_nhan}
    }
    tieu_de = {"Content-Type": "application/json"}
    try:
        phan_hoi = requests.post(apifb_2, json=tradulieu, headers=tieu_de)
        print(f"Trang thai gui tin nhan: {phan_hoi.status_code}")
    except Exception as e:
        print(f"Loi gui tin nhan: {e}")

def xu_ly_va_tra_loi(id_nguoi_gui: str, noi_dung_tin_nhan: str):
    """Ham chay ngam: Dua cau hoi vao RAG va gui cau tra loi"""
    try:
        # Gọi chuỗi hoàn chỉnh kèm truyền ID Facebook làm khóa bộ nhớ
        cau_tra_loi = chuoi_hoan_chinh.invoke(
            {"cau_hoi": noi_dung_tin_nhan},
            config={"configurable": {"session_id": id_nguoi_gui}}
        )
        
        # Lam sach ky tu Markdown truoc khi gui qua Messenger
        cau_tra_loi_sach = lam_sach_markdown(cau_tra_loi)
        
        # Gui ket qua da lam sach ve Facebook
        gui_tin_nhan_facebook(id_nguoi_gui, cau_tra_loi_sach)
    except Exception as e:
        gui_tin_nhan_facebook(id_nguoi_gui, "He thong tu van dang qua tai, ban vui long thu lai sau it phut nhe.")
        print(f"Loi RAG: {e}")

@app.get("/webhook")
async def xac_minh_webhook_facebook(yeu_cau: Request):
    """Facebook goi cong nay de xac minh ket noi"""
    tham_so = yeu_cau.query_params
    if tham_so.get("hub.mode") == "subscribe" and tham_so.get("hub.verify_token") == xacminh:
        return Response(content=tham_so.get("hub.challenge"), media_type="text/plain")
    return Response(content="Xac thuc that bai", status_code=403)

@app.post("/webhook")
async def nhan_tin_nhan_facebook(yeu_cau: Request, tac_vu_nen: BackgroundTasks):
    """Facebook goi cong nay moi khi co nguoi nhan tin vao Fanpage"""
    noi_dung_yeu_cau = await yeu_cau.json()
    
    # Kiem tra xem co dung la tin nhan tu Page khong
    if noi_dung_yeu_cau.get("object") == "page":
        for muc_du_lieu in noi_dung_yeu_cau.get("entry", []):
            for su_kien_tin_nhan in muc_du_lieu.get("messaging", []):
                # Neu la su kien nhan tin nhan van ban
                if su_kien_tin_nhan.get("message") and su_kien_tin_nhan["message"].get("text"):
                    id_nguoi_gui = su_kien_tin_nhan["sender"]["id"]
                    tin_nhan_khach = su_kien_tin_nhan["message"]["text"]
                    
                    print(f"📩 Khach [{id_nguoi_gui}] hoi: {tin_nhan_khach}")
                    
                    # Nem task cho RAG xu ly ngam de tra ngay ma 200 cho Facebook
                    tac_vu_nen.add_task(xu_ly_va_tra_loi, id_nguoi_gui, tin_nhan_khach)
                    
    # Luon phai tra ve "EVENT_RECEIVED" kem ma 200 OK ngay lap tuc
    return Response(content="EVENT_RECEIVED", status_code=200)

@app.get("/")
async def trang_chu():
    return {"message": "Server FastAPI Chatbot Dai Nam dang hoat dong!"}