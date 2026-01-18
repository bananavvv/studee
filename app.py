import streamlit as st
import os
import tempfile
import re
import random
import pandas as pd
from typing import List, Dict, Any
from dotenv import load_dotenv

# AI & Processing Libraries
from llama_parse import LlamaParse
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Studee Pro",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)
load_dotenv()


# --- 2. CSS STYLING ---
def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .main-header { font-size: 3rem; font-weight: 700; color: #006C85; margin-bottom: 0; }

        /* FLASHCARD TEXT FIX */
        .flashcard-container p, .flashcard-text {
            color: #000000 !important;
            font-size: 1.8rem;
            font-weight: 600;
            line-height: 1.5;
            margin: 0;
        }

        /* BUTTONS */
        div.stButton > button {
            border-radius: 10px;
            font-weight: 600;
            border: 1px solid #ddd;
            height: 3em;
        }
        div.stButton > button:hover {
            border-color: #006C85;
            color: #006C85;
        }
        button[kind="primary"] { background-color: #006C85; border: none; }

        /* QUIZ */
        .stRadio [role=radiogroup] { padding: 10px; background: transparent; }
    </style>
    """, unsafe_allow_html=True)


def inject_confetti():
    st.markdown("""
        <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js"></script>
        <script>
            confetti({particleCount: 150, spread: 70, origin: { y: 0.6 }});
        </script>
    """, unsafe_allow_html=True)


# --- 3. LOGIC CLASSES ---

class DocumentProcessor:
    def __init__(self, groq_key: str, llama_key: str):
        self.groq_key = groq_key
        self.llama_key = llama_key

    @st.cache_resource(show_spinner=False)
    def process_files(_self, uploaded_files: List[Any]) -> FAISS:
        all_documents = []
        parser = LlamaParse(api_key=_self.llama_key, result_type="markdown", verbose=True)

        for uploaded_file in uploaded_files:
            file_ext = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                docs = parser.load_data(tmp_path)
                for doc in docs:
                    doc.metadata = {"source": uploaded_file.name}
                all_documents.extend(docs)
            finally:
                os.remove(tmp_path)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        lc_docs = [Document(page_content=d.text, metadata=d.metadata) for d in all_documents]
        chunks = text_splitter.split_documents(lc_docs)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.from_documents(chunks, embeddings)


class LLMEngine:
    def __init__(self, api_key: str, temperature: float = 0.3):
        self.llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant", temperature=temperature)

    def query(self, vector_store: FAISS, prompt_text: str, k: int = 10) -> str:
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": 50})
        docs = retriever.invoke(prompt_text)
        context = "\n\n".join([d.page_content for d in docs])

        full_prompt = ChatPromptTemplate.from_template("""
        Role: Expert AI Tutor.
        Goal: Answer based ONLY on context.
        Context: {context}
        Query: {input}
        """)

        chain = full_prompt | self.llm
        return chain.invoke({"context": context, "input": prompt_text}).content


class ContentGenerator:
    @staticmethod
    def clean_text(text: str) -> str:
        text = text.replace("**", "").strip()
        text = re.sub(r'^(?:Q\d+[:\.]?|Question\s*\d*[:\.]?|\d+[\.\)\:])\s*', '', text, flags=re.IGNORECASE)
        return text.strip()

    @staticmethod
    def parse_quiz(raw_text: str) -> List[Dict]:
        questions = []
        lines = raw_text.strip().split("\n")

        for line in lines:
            if "|||" in line:
                parts = line.split("|||")
                if len(parts) >= 6:
                    try:
                        raw_q = parts[0].strip()
                        clean_q = ContentGenerator.clean_text(raw_q)
                        if len(clean_q) < 5: continue

                        options = [ContentGenerator.clean_text(p) for p in parts[1:5]]
                        ans_idx = int(parts[5].strip())
                        explanation = parts[6].strip() if len(parts) > 6 else "Check notes."

                        if 0 <= ans_idx < len(options):
                            correct_txt = options[ans_idx]
                            random.shuffle(options)
                            new_idx = options.index(correct_txt)
                            questions.append({
                                "question": clean_q, "options": options,
                                "correct": new_idx, "explanation": explanation
                            })
                    except:
                        continue
        return questions

    @staticmethod
    def parse_flashcards(raw_text: str) -> List[Dict]:
        cards = []
        lines = raw_text.strip().split("\n")
        for line in lines:
            if "|||" in line:
                parts = line.split("|||")
                if len(parts) >= 2:
                    front = ContentGenerator.clean_text(parts[0])
                    back = ContentGenerator.clean_text(parts[1])

                    # --- JUNK FILTER ---
                    # Skips lines that are just headers like "Term ||| Definition"
                    if front.lower() in ["term", "concept", "front", "word", "question"]:
                        continue

                    if len(front) > 1 and len(back) > 2:
                        cards.append({"front": front, "back": back})
        return cards


# --- 4. MAIN APP ---

def main():
    inject_custom_css()

    defaults = {
        "vector_store": None, "chat_history": [], "quiz_data": [],
        "flashcards": [], "fc_index": 0, "fc_flipped": False,
        "last_files": [], "outcomes": None
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

    if not isinstance(st.session_state.fc_index, int):
        st.session_state.fc_index = 0

    with st.sidebar:
        if os.path.exists("logo.png"):
            st.image("logo.png", width=200)
        else:
            st.title("📘 STUDEE PRO")
        st.write("---")

        with st.expander("⚙️ Settings"):
            creativity = st.slider("Creativity", 0.0, 1.0, 0.3)

        groq_key = os.getenv("GROQ_API_KEY")
        llama_key = os.getenv("LLAMA_CLOUD_API_KEY")

        if not groq_key or not llama_key:
            st.error("🔑 Secrets Missing!")
            st.stop()

        st.subheader("📂 Upload Notes")
        files = st.file_uploader("PDF, DOCX, PPTX", type=["pdf", "docx", "pptx"], accept_multiple_files=True)

        current_names = [f.name for f in files] if files else []
        last_names = [f.name for f in st.session_state.last_files]
        if current_names != last_names:
            st.session_state.clear()
            st.session_state.last_files = files
            st.rerun()

        if files and not st.session_state.vector_store:
            if st.button("🚀 Analyze", type="primary", use_container_width=True):
                processor = DocumentProcessor(groq_key, llama_key)
                llm_engine = LLMEngine(groq_key, temperature=creativity)

                with st.status("Processing...", expanded=True) as status:
                    vs = processor.process_files(files)
                    st.session_state.vector_store = vs
                    outcomes = llm_engine.query(vs, "List 5 key learning outcomes. Format: * 🎯 [Outcome]")
                    st.session_state.outcomes = outcomes
                    status.update(label="Complete!", state="complete", expanded=False)
                st.rerun()

    if st.session_state.vector_store:
        c1, c2 = st.columns([3, 1])
        c1.markdown('<h1 class="main-header">Studee Dashboard</h1>', unsafe_allow_html=True)

        if st.session_state.outcomes:
            with st.expander("🎯 Learning Outcomes", expanded=False):
                st.markdown(st.session_state.outcomes)

        llm_engine = LLMEngine(groq_key, temperature=creativity)
        t1, t2, t3 = st.tabs(["💬 Chat", "📝 Quiz", "🎴 Flashcards"])

        # === CHAT ===
        with t1:
            for msg in st.session_state.chat_history:
                avatar = "👨‍🎓" if msg["role"] == "user" else "🤖"
                st.chat_message(msg["role"], avatar=avatar).markdown(msg["content"])

            if query := st.chat_input("Ask a question..."):
                st.session_state.chat_history.append({"role": "user", "content": query})
                st.chat_message("user", avatar="👨‍🎓").markdown(query)

                with st.chat_message("assistant", avatar="🤖"):
                    response = llm_engine.query(st.session_state.vector_store, query)
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})

        # === QUIZ ===
        with t2:
            c_a, c_b = st.columns([3, 1])
            c_a.markdown("#### 📝 Exam Simulator")
            if c_b.button("🔄 New Quiz", use_container_width=True):
                st.session_state.quiz_data = []
                st.rerun()

            if not st.session_state.quiz_data:
                with st.spinner("Generating 10 unique questions..."):
                    prompt = """
                    Create 7 multiple choice questions.
                    Rules: 
                    1. Options must be descriptive (10-15 words).
                    2. Do NOT use markdown bolding in questions or options.
                    3. Do NOT number the questions.
                    4. Start directly with the text.
                    Format: Question Text Here ||| OpA ||| OpB ||| OpC ||| OpD ||| CorrectIdx(0-3) ||| Explanation
                    """
                    q1 = llm_engine.query(st.session_state.vector_store, "Introductory concepts " + prompt, k=15)
                    q2 = llm_engine.query(st.session_state.vector_store, "Advanced details " + prompt, k=15)

                    parsed = ContentGenerator.parse_quiz(q1) + ContentGenerator.parse_quiz(q2)

                    if parsed:
                        st.session_state.quiz_data = parsed[:10]
                        st.rerun()
                    else:
                        st.error("AI Generation Failed. Please try clicking 'New Quiz' again.")

            if st.session_state.quiz_data:
                csv = pd.DataFrame(st.session_state.quiz_data).to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download CSV", csv, "quiz.csv", "text/csv")

                with st.form("quiz_form"):
                    answers = {}
                    for i, q in enumerate(st.session_state.quiz_data):
                        st.markdown(f"**{i + 1}. {q['question']}**")
                        answers[i] = st.radio("Select:", q['options'], key=f"q_{i}", index=None,
                                              label_visibility="collapsed")
                        st.divider()

                    if st.form_submit_button("Submit Exam", type="primary", use_container_width=True):
                        score = sum([1 for i, q in enumerate(st.session_state.quiz_data) if
                                     answers.get(i) == q['options'][q['correct']]])
                        if score == len(st.session_state.quiz_data):
                            st.balloons()
                            inject_confetti()
                            st.success(f"🏆 Perfect Score: {score}/{len(st.session_state.quiz_data)}")
                        else:
                            st.warning(f"Score: {score}/{len(st.session_state.quiz_data)}")
                            for i, q in enumerate(st.session_state.quiz_data):
                                if answers.get(i) != q['options'][q['correct']]:
                                    st.error(f"Q{i + 1}: {q['question']}")
                                    st.info(f"Correct: {q['options'][q['correct']]} \n\nReason: {q['explanation']}")

        # === FLASHCARDS ===
        with t3:
            c_a, c_b = st.columns([3, 1])
            c_a.markdown("#### 🎴 Flashcards")
            if c_b.button("🔄 New Deck", use_container_width=True):
                st.session_state.flashcards = []
                st.rerun()

            if not st.session_state.flashcards:
                with st.spinner("Creating deck..."):
                    prompt = """
                    Create 7 descriptive flashcards.
                    RULES:
                    1. Do NOT output a header row like 'Term ||| Definition'.
                    2. Start directly with the first term.
                    3. Do not number the cards.
                    Format: Term ||| Definition (2 sentences)
                    """
                    raw1 = llm_engine.query(st.session_state.vector_store, "Basic terms " + prompt, k=15)
                    raw2 = llm_engine.query(st.session_state.vector_store, "Advanced concepts " + prompt, k=15)
                    st.session_state.flashcards = (
                                ContentGenerator.parse_flashcards(raw1) + ContentGenerator.parse_flashcards(raw2))[:10]
                    st.rerun()

            if st.session_state.flashcards:
                total = len(st.session_state.flashcards)
                if st.session_state.fc_index >= total:
                    st.session_state.fc_index = total - 1

                idx = st.session_state.fc_index
                card = st.session_state.flashcards[idx]

                st.progress((idx + 1) / total)
                st.caption(f"Card {idx + 1} of {total}")

                border_color = "#26C6DA" if st.session_state.fc_flipped else "#006C85"
                lbl = "DEFINITION" if st.session_state.fc_flipped else "TERM"
                content = card['back'] if st.session_state.fc_flipped else card['front']

                # --- FIXED HTML with CLASS-BASED CSS OVERRIDE ---
                st.markdown(f"""
                <div class="flashcard-container" style="
                    background-color: white; 
                    border-radius: 20px; 
                    padding: 50px; 
                    min-height: 300px; 
                    display: flex; 
                    flex-direction: column; 
                    align-items: center; 
                    justify-content: center;
                    text-align: center;
                    border: 2px solid #e0e0e0;
                    border-left: 10px solid {border_color};
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                ">
                    <div style="
                        font-size: 14px; 
                        color: {border_color}; 
                        font-weight: bold; 
                        letter-spacing: 2px; 
                        margin-bottom: 20px; 
                        text-transform: uppercase;">
                        {lbl}
                    </div>
                    <div class="flashcard-text">
                        <p>{content}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                c1, c2, c3 = st.columns([1, 2, 1])
                if c1.button("⬅️ Previous", use_container_width=True):
                    if idx > 0:
                        st.session_state.fc_index -= 1
                        st.session_state.fc_flipped = False
                        st.rerun()
                if c2.button("🔄 Flip", type="primary", use_container_width=True):
                    st.session_state.fc_flipped = not st.session_state.fc_flipped
                    st.rerun()
                if c3.button("Next ➡️", use_container_width=True):
                    if idx < total - 1:
                        st.session_state.fc_index += 1
                        st.session_state.fc_flipped = False
                        st.rerun()
                    else:
                        st.success("End of Deck!")

    else:
        st.markdown('<div style="text-align: center; padding: 50px;">', unsafe_allow_html=True)
        if os.path.exists("logo.png"): st.image("logo.png", width=120)
        st.markdown('<h1 class="main-header">Welcome to Studee Pro</h1>', unsafe_allow_html=True)
        st.info("Upload documents in the sidebar to begin.")


if __name__ == "__main__":
    main()
