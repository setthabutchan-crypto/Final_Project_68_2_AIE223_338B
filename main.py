import os
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Load Environment Variables (ตั้งค่า API Key)
# แนะนำให้สร้างไฟล์ .env และใส่ GOOGLE_API_KEY=your_api_key_here
load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", # เปลี่ยนชื่อโมเดลที่นี่
    version="v1"             # ระบุเวอร์ชันให้ชัดเจน
)


# 2. Define State (โครงสร้างข้อมูลของระบบ)
class EmailState(TypedDict):
    original_text: str
    category: str
    rag_context: str
    draft_response: str

# 3. Agent 1: Categorization Node
def categorize_email(state: EmailState):
    print("กำลังทำงาน: [Agent 1] วิเคราะห์และจัดหมวดหมู่...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "คุณคือผู้เชี่ยวชาญด้าน Customer Support จัดหมวดหมู่อีเมลเป็น 1 ใน 4 หมวด: [Billing, Tech_Support, Inquiry, Spam] ห้ามตอบคำอื่นนอกจากชื่อหมวดหมู่"),
        ("user", "อีเมลลูกค้า: {original_text}")
    ])
    chain = prompt | llm
    response = chain.invoke({"original_text": state['original_text']})
    return {"category": response.content.strip()}

# 4. RAG Node: Mock Data Retrieval
def retrieve_knowledge(state: EmailState):
    print(f"กำลังทำงาน: [System] ดึงข้อมูลจากฐานข้อมูลสำหรับหมวดหมู่: {state['category']}...")
    # ใน Production จริง ตรงนี้คือการใช้ Vector Store (เช่น Pinecone, FAISS)
    mock_db = {
        "Billing": "นโยบาย: ลูกค้าสามารถขอคืนเงินได้ภายใน 7 วันหลังจากการซื้อ แนบลิงก์ refund.com/apply",
        "Tech_Support": "วิธีแก้ปัญหา: ให้ลูกค้ารีสตาร์ทเราเตอร์ หากไม่หายให้แจ้ง MAC Address มาที่ทีมวิศวกร",
        "Inquiry": "ตอบกลับทั่วไป: ขอบคุณที่สนใจบริการของเรา กรุณาดูแคตตาล็อกที่เว็บไซต์หลัก",
        "Spam": "ไม่จำเป็นต้องตอบกลับ"
    }
    context = mock_db.get(state['category'], "ไม่มีข้อมูลในระบบ")
    return {"rag_context": context}

# 5. Agent 2: Drafting Node
def draft_response(state: EmailState):
    print("กำลังทำงาน: [Agent 2] กำลังร่างอีเมลตอบกลับ...")
    
    # หากเป็น Spam ให้ข้ามการร่างอีเมล
    if state['category'] == "Spam":
        return {"draft_response": "[ระบบข้ามการตอบกลับเนื่องจากถูกจัดหมวดหมู่เป็น Spam]"}

    prompt = ChatPromptTemplate.from_messages([
        ("system", """คุณคือ Customer Success Expert ร่างอีเมลตอบกลับลูกค้าด้วยความสุภาพและเป็นมืออาชีพ 
        ใช้ข้อมูลอ้างอิงต่อไปนี้เท่านั้นในการร่างคำตอบ: {rag_context}
        ลงท้ายอีเมลด้วย 'ทีมงาน Customer Support'"""),
        ("user", "อีเมลลูกค้า: {original_text}")
    ])
    chain = prompt | llm
    response = chain.invoke({
        "original_text": state['original_text'],
        "rag_context": state['rag_context']
    })
    return {"draft_response": response.content}

# 6. Build and Compile Graph
workflow = StateGraph(EmailState)

workflow.add_node("categorizer", categorize_email)
workflow.add_node("retriever", retrieve_knowledge)
workflow.add_node("drafter", draft_response)

workflow.set_entry_point("categorizer")
workflow.add_edge("categorizer", "retriever")
workflow.add_edge("retriever", "drafter")
workflow.add_edge("drafter", END)

app = workflow.compile()

# ==========================================
# 7. Execution Block (จุดสำหรับรันทดสอบ)
# ==========================================
if __name__ == "__main__":
    print("=== เริ่มต้นระบบ AI Email Agent ===")
    
    # จำลองอีเมลที่ลูกค้าส่งมา
    sample_email = "สวัสดีครับ ผมเพิ่งตัดบัตรเครดิตซื้อแพ็กเกจรายปีไปเมื่อวาน แต่เปลี่ยนใจอยากยกเลิกและขอเงินคืน ไม่ทราบว่าทำได้ไหมครับ?"
    print(f"อีเมลเข้า: '{sample_email}'\n")
    
    # รัน Workflow
    initial_state = {"original_text": sample_email}
    result = app.invoke(initial_state)
    
    print("\n=== สรุปผลลัพธ์ (Final Output) ===")
    print(f"หมวดหมู่ที่จัดได้: {result['category']}")
    print(f"ข้อมูล RAG ที่ดึงมา: {result['rag_context']}")
    print(f"ฉบับร่างอีเมล:\n{result['draft_response']}")
    print("====================================")