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
import datetime # Zaman damgası için eklendi

# --- 1. AYARLAR ---
warnings.filterwarnings("ignore")
st.set_page_config(page_title="ReAct Final Projesi", layout="wide", page_icon="🚀")
load_dotenv()

# Hızlı ve İtaatkar Model
MODEL_ID = "llama-3.1-8b-instant" 

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

# --- 2. RAG MOTORU (Hızlandırılmış) ---
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
                    # Çöp sayfaları atla (İçindekiler, çok kısa sayfalar)
                    if len(text.split()) < 50 or "CONTENTS" in text[:50].upper(): continue
                    
                    tokens = text_splitter.encode(text)
                    # Chunk boyutunu optimize ettim: 500 token
                    for j in range(0, len(tokens), 500):
                        chunk = text_splitter.decode(tokens[j:j+500])
                        all_docs.append(chunk)
                        all_metas.append({"source": os.path.basename(path), "page": i + 1})
                        all_ids.append(f"{os.path.basename(path)}_{i}_{j}")
        
        if all_docs:
            batch_size = 128 # Daha hızlı yükleme için batch artırıldı
            for i in range(0, len(all_docs), batch_size):
                end = min(i + batch_size, len(all_docs))
                self.collection.add(
                    documents=all_docs[i:end],
                    embeddings=self.embedding_model.encode(all_docs[i:end]).tolist(),
                    metadatas=all_metas[i:end],
                    ids=all_ids[i:end]
                )
            status.update(label="Sistem Hazır", state="complete", expanded=False)

    def search(self, query: str, top_k: int = 3) -> str:
        if self.collection.count() == 0: return "Veritabanı boş."
        results = self.collection.query(
            query_embeddings=[self.embedding_model.encode(query).tolist()],
            n_results=top_k
        )
        if not results['documents'][0]: return "Dokümanlarda bilgi bulunamadı."
        
        # Sadece en alakalı kısımları birleştir
        context = ""
        for i, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i]
            context += f"\n[Kaynak: {meta['source']}, Sayfa: {meta['page']}]\n{doc[:1000]}..." # Metni kırp
        return context

# --- 3. ARAÇLAR (Kuvvetlendirilmiş) ---
class ToolBox:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.ddgs = DDGS()

    def search_docs(self, query: str) -> str:
        """Dokümanlarda arama yapar."""
        return self.kb.search(query)

    def web_search(self, query: str) -> str:
        """İnternette arama yapar (Gelişmiş)."""
        try:
            # max_results 4'e çıkarıldı, daha fazla veri
            results = self.ddgs.text(query, max_results=4)
            if not results: return "Sonuç bulunamadı."
            # Ajanın okuması için temiz format
            return "\n".join([f"Başlık: {r['title']}\nÖzet: {r['body']}" for r in results])
        except Exception as e: return f"Web Hatası: {e}"

    def calculator(self, expression: str) -> str:
        """Matematiksel işlem yapar."""
        try:
            return str(eval(expression, {"__builtins__": None}, {}))
        except: return "Hesaplama Hatası."

    def execute(self, name: str, input_str: str) -> str:
        name = name.strip().lower()
        if name == "search_docs": return self.search_docs(input_str)
        if name == "web_search": return self.web_search(input_str)
        if name == "calculator": return self.calculator(input_str)
        return "Bilinmeyen araç."

    def get_descriptions(self) -> str:
        return """
1. search_docs: Teknik PDF dokümanlarını arar. (Örn: "OpenCV imread parameters")
2. web_search: İnternette güncel bilgi arar. (Örn: "Requests library timeout default")
3. calculator: Hesaplama yapar. (Örn: "150 * 10")
"""

# --- 4. REACT AJAN (Optimize Edilmiş Beyin) ---
class ReActAgent:
    def __init__(self, api_key: str, toolbox: ToolBox):
        self.client = Groq(api_key=api_key)
        self.toolbox = toolbox
        self.action_re = re.compile(r'^Action: (\w+): (.*)$') 

    def run(self, question: str, chat_history: list):
        memory = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-2:]])
        
        system_prompt = f"""
        Sen Python Veri Bilimi alanında uzman bir ReAct (Reasoning + Acting) Ajanısın.
        Görevin: Kullanıcı sorularına elindeki araçları kullanarak adım adım, mantıklı ve doğru cevaplar vermektir.

        MEVCUT ARAÇLARIN:
        {self.toolbox.get_descriptions()}

        TAKİP ETMEN GEREKEN FORMAT (BU YAPIYI KESİNLİKLE BOZMA):
        Question: Kullanıcının sorusu
        Thought: Soruyu çözmek için ne yapmalıyım? Hangi aracı kullanmalıyım? (Her zaman Türkçe düşün)
        Action: [Araç Adı]: [Girdi]
        Observation: Aracın çıktısı (Sistem tarafından sağlanır)
        ... (Gerekirse tekrar Düşün ve Aksiyon al) ...
        Answer: Nihai cevap (Bulduğun bilgiyi Türkçe yaz)

        ÖRNEK OTURUM:
        Question: Requests kütüphanesinin varsayılan timeout süresi nedir?
        Thought: Bu teknik bir Python sorusu. Önce 'search_docs' aracını kullanarak Requests dökümantasyonunu taramalıyım.
        Action: search_docs: requests default timeout
        Observation: [Kaynak: requests.pdf] ...timeout varsayılan olarak None değerindedir, yani bir zaman aşımı yoktur...
        Thought: Bilgiyi dökümanda buldum. Varsayılan değer 'None'. Başka bir işlem yapmama gerek yok.
        Answer: Requests kütüphanesinde varsayılan timeout süresi 'None'dır, yani varsayılan olarak bir zaman aşımı yoktur.

        KRİTİK KURALLAR (KESİNLİKLE UY):
        1. **STRATEJİ:** Önce `search_docs` ile dokümanları tara. Eğer dokümanlarda net bir cevap bulamazsan İNATLAŞMA, hemen `web_search` aracını kullan.
        2. **DÖNGÜ KORUMASI:** Eğer Observation kısmında "BU BİLGİYİ ZATEN ALDIN" uyarısını görürsen, ASLA aynı aramayı tekrar yapma. Hemen elindeki bilgiyle veya genel bilginle 'Answer:' yazıp bitir.
        3. **CEVAPLAMA:** Cevabı bulduğun an (Observation tatmin ediciyse) daha fazla arama yapma, hemen `Answer:` formatında cevabı ver.
        4. **DİL:** Düşüncelerin ve Cevapların HEP TÜRKÇE olsun.

        Soru: {question}
        """.strip()

        scratchpad = system_prompt
        trace_log = [] 
        used_actions = set()

        step_count = 0
        while step_count < 7:
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

            # --- 2. AKSİYON AYRIŞTIRMA ---
            actions = [self.action_re.match(a) for a in result.split('\n') if self.action_re.match(a)]
            
            if actions:
                action, action_input = actions[0].groups()
                
                # --- AKILLI DÖNGÜ KIRICI ---
                action_key = f"{action}:{action_input.strip()}"
                if action_key in used_actions:
                    # Model aynı şeyi yaparsa, ona kızmıyoruz, cevabı yazmaya zorluyoruz
                    observation = "HATA: Aynı aramayı tekrar yapıyorsun! Bu yasaktır. Lütfen ya farklı bir araç dene (örn: web_search) ya da bildiklerinle 'Answer:' diyerek cevabı yaz."
                else:
                    used_actions.add(action_key)
                    yield {"type": "action", "tool": action, "input": action_input}
                    observation = self.toolbox.execute(action, action_input)
                
                obs_log = f"\nObservation: {observation}\n"
                scratchpad += obs_log
                trace_log.append(f"Observation: {observation}")
                
                yield {"type": "observation", "content": "Veri alındı."}
            else:
                # Model saçmalarsa uyar
                scratchpad += "\nObservation: Lütfen bir Aksiyon al veya 'Answer:' ile bitir.\n"

        yield {"type": "final", "content": "Adım limiti doldu.", "trace": trace_log}

# --- 5. ARAYÜZ (Modern ve Temiz) ---
def main():
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("⚡ Hızlı ReAct Ajanı")
        
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

        st.markdown("---")
        st.markdown("**Test Soruları:**")
        if st.button("Test 1: OpenCV Nedir?"):
            process_input("OpenCV kütüphanesi ne işe yarar?")
        if st.button("Test 2: Multi-Hop Hesap"):
            process_input("Requests kütüphanesinin varsayılan timeout süresini bul ve 20 ile çarp.")

    st.title("🤖 Final Ödev Ajanı (V3)")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Bir soru sorun..."):
        process_input(prompt)

def process_input(prompt):
    if not st.session_state.agent:
        st.warning("Lütfen sistemi başlatın.")
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("Analiz ediliyor...", expanded=True)
        response_box = st.empty()
        full_trace = []
        final_res = ""

        for step in st.session_state.agent.run(prompt, st.session_state.messages):
            if step["type"] == "action":
                status.write(f"⚙️ **İşlem:** `{step['tool']}` aranıyor...")
            elif step["type"] == "observation":
                status.write("✅ Veri Bulundu")
            elif step["type"] == "final":
                final_res = step["content"]
                full_trace = step["trace"]
                status.update(label="Tamamlandı", state="complete", expanded=False)
            elif step["type"] == "error":
                st.error(step["content"])
        
        if final_res:
            response_box.markdown(final_res)
            st.session_state.messages.append({"role": "assistant", "content": final_res})
            
            # --- LOG KAYDETME (.LOG UZANTISIYLA) ---
            try:
                # 'agent_trace.log' dosyasına ekleme modu ('a') ile yazıyoruz
                with open("agent_trace.log", "a", encoding="utf-8") as f:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"\n{'='*50}\n")
                    f.write(f"ZAMAN: {timestamp}\n")
                    f.write(f"SORU: {prompt}\n")
                    f.write(f"{'='*50}\n")
                    f.write("\n".join(full_trace))
                    f.write(f"\n\nCEVAP: {final_res}\n")
                    f.write(f"{'-'*50}\n")
                
                st.toast("Düşünce zinciri 'agent_trace.log' dosyasına kaydedildi.", icon="💾")
            except Exception as e:
                st.error(f"Log kaydetme hatası: {e}")
            # -------------------------------------------

            with st.expander("📝 Rapor İçin Trace (Kopyala)"):
                st.code("\n".join(full_trace), language="text")

if __name__ == "__main__":
    main()