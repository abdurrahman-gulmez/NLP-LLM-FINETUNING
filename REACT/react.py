import os
import re
import sys
import math
import warnings
import streamlit as st
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS
import tiktoken
from groq import Groq
import fitz  # PyMuPDF
import datetime

# --- 1. AYARLAR ---
warnings.filterwarnings("ignore")
st.set_page_config(page_title="ReAct Final Projesi", layout="wide", page_icon="🧠")
load_dotenv()

# KRİTİK DEĞİŞİKLİK: Multi-hop mantığı için 70B modeli şarttır. 
# 8B modeli ikinci adımda unutkanlık yapar.
MODEL_ID = "llama-3.3-70b-versatile" 

PDF_PATHS = [
    "docs/beautiful-soup-4-readthedocs-io-en-latest.pdf",
    "docs/Matplotlib.pdf",
    "docs/numpy-ref.pdf",
    "docs/opencv_tutorials.pdf",
    "docs/requests-readthedocs-io-en-latest.pdf",
    "docs/scikit-learn-docs.pdf",
    "docs/pillow-readthedocs-io-en-latest.pdf",
    "docs/pymupdf-readthedocs-io-en-latest.pdf",
    "docs/xgboost-readthedocs-io-en-latest.pdf"
]

# --- 2. RAG MOTORU (Temizlenmiş Çıktı) ---
class KnowledgeBase:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.collection = self.client.get_or_create_collection(name="final_kb")

    def ingest_documents(self):
        if self.collection.count() > 0: return
        
        status = st.status("Veri Tabanı İndeksleniyor...", expanded=True)
        text_splitter = tiktoken.get_encoding("cl100k_base")
        all_docs, all_metas, all_ids = [], [], []
        
        for path in PDF_PATHS:
            if not os.path.exists(path): continue
            status.write(f"Okunuyor: {os.path.basename(path)}")
            with fitz.open(path) as doc:
                for i, page in enumerate(doc):
                    text = page.get_text()
                    if len(text.split()) < 50 or "CONTENTS" in text[:50].upper(): continue
                    
                    tokens = text_splitter.encode(text)
                    for j in range(0, len(tokens), 500):
                        chunk = text_splitter.decode(tokens[j:j+500])
                        all_docs.append(chunk)
                        all_metas.append({"source": os.path.basename(path), "page": i + 1})
                        all_ids.append(f"{os.path.basename(path)}_{i}_{j}")
        
        if all_docs:
            batch_size = 128
            for i in range(0, len(all_docs), batch_size):
                end = min(i + batch_size, len(all_docs))
                self.collection.add(
                    documents=all_docs[i:end],
                    embeddings=self.embedding_model.encode(all_docs[i:end]).tolist(),
                    metadatas=all_metas[i:end],
                    ids=all_ids[i:end]
                )
            status.update(label="Sistem Hazır", state="complete", expanded=False)

    def search(self, query: str, top_k: int = 2) -> str:
        if self.collection.count() == 0: return "Veritabanı boş."
        results = self.collection.query(
            query_embeddings=[self.embedding_model.encode(query).tolist()],
            n_results=top_k
        )
        if not results['documents'][0]: return "Dokümanlarda bilgi bulunamadı."
        
        context = ""
        for i, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i]
            # KRİTİK: Satır sonlarını boşlukla değiştiriyoruz.
            # Modelin kafası karışmasın diye metni tek satıra indiriyoruz.
            clean_doc = doc.replace("\n", " ").replace("  ", " ")
            context += f"\n[Kaynak: {meta['source']}, Sayfa: {meta['page']}] İÇERİK: {clean_doc[:500]}..." 
        return context

# --- 3. ARAÇLAR ---
class ToolBox:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.ddgs = DDGS()

    def search_docs(self, query: str) -> str:
        return self.kb.search(query)

    def web_search(self, query: str) -> str:
        try:
            # Multi-hop için daha fazla sonuç gerekebilir ama özet kısa olmalı
            results = self.ddgs.text(query, max_results=3, timelimit='y')
            if not results: return "Sonuç bulunamadı."
            
            formatted_res = []
            for r in results:
                # Başlık ve özet temizliği
                clean_body = r['body'].replace("\n", " ")
                formatted_res.append(f"Başlık: {r['title']} | Bilgi: {clean_body}")
            
            return "\n".join(formatted_res)
        except Exception as e: return f"Hata: {e}"

    def calculator(self, expression: str) -> str:
        try:
            # Güvenli eval
            clean_expr = expression.replace("x", "*").replace(",", ".")
            return str(eval(clean_expr, {"__builtins__": None}, {}))
        except: return "Hesaplama Hatası."

    def execute(self, name: str, input_str: str) -> str:
        name = name.strip().lower()
        if name == "search_docs": return self.search_docs(input_str)
        if name == "web_search": return self.web_search(input_str)
        if name == "calculator": return self.calculator(input_str)
        return "Bilinmeyen araç."

    def get_descriptions(self) -> str:
        return """
1. search_docs: Teknik PDF dokümanlarını arar.
2. web_search: İnternet araması. Sadece ANAHTAR KELİME gir (Örn: "python requests timeout").
3. calculator: Hesaplama yapar (Örn: "10 * 5").
"""

# --- 4. REACT AJAN ---
class ReActAgent:
    def __init__(self, api_key: str, toolbox: ToolBox):
        self.client = Groq(api_key=api_key)
        self.toolbox = toolbox
        self.action_re = re.compile(r'^Action: (\w+): (.*)$') 

    def run(self, question: str, chat_history: list):
        memory = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-2:]])
        
        system_prompt = f"""
Sen Python Veri Bilimi uzmanı bir ReAct Ajanısın.
Görevin: Soruları adım adım mantık yürüterek çözmek.

MEVCUT ARAÇLAR:
{self.toolbox.get_descriptions()}

STRATEJİ (KESİNLİKLE UY):
1. **DÜŞÜN (Thought):** Her adımdan önce ne yapacağını planla.
2. **ÇOK ADIMLI (Multi-Hop):** Eğer soru "X'i bul ve Y yap" diyorsa:
   - Önce X'i bul (Action: search_docs...)
   - Sonra gelen veriyi oku.
   - Sonra Y işlemini yap (Action: calculator...)
3. **DURMA:** 'Observation' aldıktan sonra hemen yeni bir 'Thought' üret.
4. **DİL:** Hep Türkçe konuş.

FORMAT:
Question: Soru
Thought: Plan
Action: [Araç]: [Girdi]
Observation: [Çıktı]
Thought: Analiz
...
Answer: Cevap

Soru: {question}
""".strip()

        scratchpad = system_prompt
        trace_log = [] 
        used_actions = set()

        step_count = 0
        while step_count < 8:
            step_count += 1
            
            try:
                completion = self.client.chat.completions.create(
                    model=MODEL_ID,
                    messages=[{"role": "user", "content": scratchpad}],
                    temperature=0.0,
                    stop=["Observation:"]
                )
                result = completion.choices[0].message.content.strip()
            except Exception as e:
                yield {"type": "error", "content": f"API: {e}"}
                return

            scratchpad += result
            clean_result = result.replace("PAUSE", "").strip()
            trace_log.append(clean_result)

            if "Answer:" in result:
                final_answer = result.split("Answer:")[-1].strip()
                yield {"type": "final", "content": final_answer, "trace": trace_log}
                return

            actions = [self.action_re.match(a) for a in result.split('\n') if self.action_re.match(a)]
            
            if actions:
                action, action_input = actions[0].groups()
                action_input = action_input.strip().strip('"')
                
                action_key = f"{action}:{action_input}"
                if action_key in used_actions:
                    observation = "BU BİLGİYİ ZATEN ALDIN. Hemen bir sonraki adıma geç."
                else:
                    used_actions.add(action_key)
                    yield {"type": "action", "tool": action, "input": action_input}
                    observation = self.toolbox.execute(action, action_input)
                
                # KRİTİK MÜDAHALE: Modele "Thought:" kelimesini biz veriyoruz.
                # Bu, modelin Observation'dan sonra durmasını engeller ve düşünmeye zorlar.
                obs_log = f"\nObservation: {observation}\nThought:" 
                scratchpad += obs_log
                
                # Loglarda güzel görünsün diye düzeltiyoruz
                trace_log.append(f"Observation: {observation}")
                trace_log.append("Thought:") 
                
                yield {"type": "observation", "content": "Veri alındı."}
            else:
                scratchpad += "\nObservation: Lütfen bir Aksiyon al veya 'Answer:' ile bitir.\n"

        yield {"type": "final", "content": "Adım limiti doldu.", "trace": trace_log}

# --- 5. ARAYÜZ ---
def main():
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("🧠 ReAct Asistanı (Pro)")
        st.caption("Model: Llama-3.3-70b (Zeki Mod)")
        
        if st.button("Sistemi Başlat / Temizle"):
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                st.error("API Key Eksik!")
                return
            
            kb = KnowledgeBase()
            kb.ingest_documents()
            toolbox = ToolBox(kb)
            st.session_state.agent = ReActAgent(api_key, toolbox)
            st.session_state.messages = []
            st.success("Aktif!")

    st.title("🤖 Python Veri Bilimi Asistanı")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Soru sor..."):
        process_input(prompt)

def process_input(prompt):
    if not st.session_state.agent:
        st.warning("Lütfen sistemi başlatın.")
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        TOOL_MAPPING = {
            "search_docs": "📚 Teknik Dökümanlar",
            "web_search": "🌐 İnternet Araması",
            "calculator": "🧮 Hesap Makinesi"
        }

        status = st.status("Analiz ediliyor...", expanded=True)
        response_box = st.empty()
        full_trace = []
        final_res = ""

        for step in st.session_state.agent.run(prompt, st.session_state.messages):
            if step["type"] == "action":
                clean_tool = TOOL_MAPPING.get(step['tool'], step['tool'])
                status.write(f"🕵️‍♂️ **Analiz:** `{clean_tool}`")
            elif step["type"] == "observation":
                status.write("✅ Veri İşlendi")
            elif step["type"] == "final":
                final_res = step["content"]
                full_trace = step["trace"]
                status.update(label="Tamamlandı", state="complete", expanded=False)
            elif step["type"] == "error":
                st.error(step["content"])
        
        if final_res:
            response_box.markdown(final_res)
            st.session_state.messages.append({"role": "assistant", "content": final_res})
            
            try:
                with open("agent_trace.log", "a", encoding="utf-8") as f:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"\n{'='*50}\nZAMAN: {timestamp}\nSORU: {prompt}\n{'='*50}\n")
                    f.write("\n".join(full_trace))
                    f.write(f"\n\nCEVAP: {final_res}\n{'-'*50}\n")
            except: pass

            with st.expander("📝 Düşünce Zinciri (Trace)"):
                st.code("\n".join(full_trace), language="text")

if __name__ == "__main__":
    main()