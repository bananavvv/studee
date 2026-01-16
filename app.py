import streamlit as st
import os
import tempfile
import re
import random
import time  # <--- FIXED: Added this missing import
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

# --- 2. ADVANCED CSS (TEAL/CYAN THEME) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header { font-size: 3rem; font-weight: 700; color: #006C85; margin-bottom: 0px; }

    /* FLASHCARD CONTAINER */
    .flashcard-container {
        background-color: #ffffff !important;
        border-radius: 20px;
        padding: 40px;
        min-height: 350px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        border: 2px solid #e0e0e0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        transition: transform 0.3s ease;
    }

    /* BUTTONS */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid #ddd;
        transition: all 0.2s;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-color: #006C85;
        color: #006C85;
    }
    button[kind="primary"] { background-color: #006C85; border: none; }

    /* QUIZ STYLING */
    .stRadio [role=radiogroup] { background-color: transparent; }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# --- 3. API CHECK ---
groq_api_key = os.getenv("GROQ_API_KEY")
llamaparse_api_key = os.getenv("LLAMA_CLOUD_API_KEY")

if not groq_api_key or not llamaparse_api_key:
    st.error("🚨 API Keys Missing! Please check your .env file.")
    st.stop()

# --- 4. SESSION STATE ---
if "vector_store" not in st.session_state: st.session_state.vector_store = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "quiz_data" not in st.session_state: st.session_state.quiz_data = []
if "flashcards" not in st.session_state: st.session_state.flashcards = []
if "fc_index" not in st.session_state: st.session_state.fc_index = 0
if "fc_flipped" not in st.session_state: st.session_state.fc_flipped = False
if "last_uploaded_files" not in st.session_state: st.session_state.last_uploaded_files = []
if "learning_outcomes" not in st.session_state: st.session_state.learning_outcomes = None


# --- 5. ROBUST PARSING ENGINE ---

class ParsingEngine:
    @staticmethod
    def clean_text(text: str) -> str:
        """Removes markdown artifacts, bolding, and numbering."""
        text = text.replace("**", "")  # Remove bold
        text = text.replace("##", "")  # Remove headers

        # Regex to remove "1.", "1)", "Q1.", "Question 1:" at start of line
        text = re.sub(r'^[\d]+[\.\)\:\-]\s*', '', text)
        text = re.sub(r'^Q[\d]+[\.\)\:\-]\s*', '', text)
        text = re.sub(r'^Question\s*[\d]+[\.\)\:\-]\s*', '', text)

        return text.strip()

    @staticmethod
    def parse_quiz(raw_text: str) -> List[Dict]:
        questions = []
        # Split by double newline to handle multi-line questions better
        blocks = raw_text.split("\n")

        # Fallback to line processing
        for line in blocks:
            if "|||" in line:
                parts = line.split("|||")
                if len(parts) >= 6:
                    try:
                        raw_q = parts[0]
                        clean_q = ParsingEngine.clean_text(raw_q)

                        # VALIDATION: Skip empty questions
                        if len(clean_q) < 5: continue

                        options = [ParsingEngine.clean_text(p) for p in parts[1:5]]

                        # Validate Index
                        try:
                            ans_idx = int(re.search(r'\d+', parts[5]).group())
                        except:
                            ans_idx = 0  # Default to A if parsing fails

                        explanation = parts[6].strip() if len(parts) > 6 else "Review notes."

                        # Randomize
                        if 0 <= ans_idx < len(options):
                            correct_txt = options[ans_idx]
                            random.shuffle(options)
                            new_idx = options.index(correct_txt)

                            questions.append({
                                "question": clean_q,
                                "options": options,
                                "correct": new_idx,
                                "explanation": explanation
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
                    front = ParsingEngine.clean_text(parts[0])
                    back = ParsingEngine.clean_text(parts[1])

                    # FILTER: Remove generic placeholders
                    if front.lower() in ["term", "concept", "word"]: continue
                    if len(front) < 2 or len(back) < 5: continue

                    cards.append({"front": front, "back": back})
        return cards


# --- 6. LOGIC FUNCTIONS ---

def process_files(uploaded_files):
    all_documents = []
    parser = LlamaParse(api_key=llamaparse_api_key, result_type="markdown", verbose=True)
    progress_bar = st.progress(0)
    total_files = len(uploaded_files)

    for idx, uploaded_file in enumerate(uploaded_files):
        file_extension = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            documents = parser.load_data(tmp_file_path)
            all_documents.extend(documents)
        finally:
            os.remove(tmp_file_path)
        progress_bar.progress((idx + 1) / total_files)

    progress_bar.empty()
    text_content = "\n".join([doc.text for doc in all_documents])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.create_documents([text_content])
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def query_llm(llm, vector_store, query, prompt_template):
    # Use MMR with higher fetch_k to ensure we get diverse parts of the document
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 15, "fetch_k": 50})
    docs = retriever.invoke(query)
    context_text = "\n\n".join([d.page_content for d in docs])
    final_prompt = prompt_template.format(context=context_text, input=query)
    response = llm.invoke(final_prompt)
    return response.content


# --- 7. MAIN APP UI ---

with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=200)
    else:
        st.header("STUDEE")

    st.markdown("### 📂 Upload Notes")
    uploaded_files = st.file_uploader("Supported: PDF, DOCX, PPTX", type=["pdf", "docx", "pptx"],
                                      accept_multiple_files=True, label_visibility="collapsed")

    # Auto-Reset Logic
    current_file_names = [f.name for f in uploaded_files] if uploaded_files else []
    last_file_names = [f.name for f in
                       st.session_state.last_uploaded_files] if st.session_state.last_uploaded_files else []

    if current_file_names != last_file_names:
        st.session_state.vector_store = None
        st.session_state.chat_history = []
        st.session_state.quiz_data = []
        st.session_state.flashcards = []
        st.session_state.learning_outcomes = None
        st.session_state.last_uploaded_files = uploaded_files

    if uploaded_files and st.session_state.vector_store is None:
        if st.button("🚀 Analyze Documents", type="primary"):
            vs = process_files(uploaded_files)
            st.session_state.vector_store = vs
            temp_llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant", temperature=0.3)
            outcome_prompt = """
            Analyze the uploaded text and list the 5-7 most important learning outcomes.
            Use bullet points with an emoji for each. Format: * 🧠 [Outcome]
            <context>{context}</context>
            """
            outcomes = query_llm(temp_llm, vs, "What are the learning outcomes?", outcome_prompt)
            st.session_state.learning_outcomes = outcomes
            st.toast("Analysis Complete! Ready to study.", icon="✅")
            st.rerun()

    st.markdown("---")
    st.caption("Studee v2.0 • Powered by Groq & LlamaParse")

# --- MAIN CONTENT ---

if st.session_state.vector_store is None:
    st.markdown('<div style="text-align: center; padding-top: 50px;">', unsafe_allow_html=True)
    if os.path.exists("logo.png"): st.image("logo.png", width=150)
    st.markdown('<h1 class="main-header">Welcome to Studee</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Your AI-Powered Revision Supercharger</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**🤖 Chat with Notes**\n\nAsk questions about your PDFs, Slides, or Word Docs instantly.")
    with c2:
        st.success("**📝 Generate Quizzes**\n\nTest yourself with AI-generated multiple choice questions.")
    with c3:
        st.warning("**🧠 Smart Flashcards**\n\nActive recall made easy with auto-generated cards.")
    st.markdown("---")
    st.markdown("<center>👈 <b>Upload a file in the sidebar to get started!</b></center>", unsafe_allow_html=True)

else:
    col_header, col_logo = st.columns([4, 1])
    with col_header:
        st.markdown('<h1 style="color:#006C85;">Studee Dashboard</h1>', unsafe_allow_html=True)
    with col_logo:
        if os.path.exists("logo.png"):
            st.image("logo.png", width=120)
        else:
            st.markdown('<div style="text-align: right; font-size: 3em;">📘</div>', unsafe_allow_html=True)

    if st.session_state.learning_outcomes:
        with st.expander("🎯 Key Learning Outcomes (Click to Expand)", expanded=False):
            st.markdown(st.session_state.learning_outcomes)

    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant", temperature=0.3)
    tab1, tab2, tab3 = st.tabs(["💬 Chat Assistant", "📝 Interactive Quiz", "🎴 Flashcards"])

    # === TAB 1: CHAT ===
    with tab1:
        st.markdown("#### 💬 Chat with your notes")
        for msg in st.session_state.chat_history:
            role_icon = "👨‍🎓" if msg["role"] == "user" else "🤖"
            st.chat_message(msg["role"], avatar=role_icon).markdown(msg["content"])

        if prompt := st.chat_input("Ask a question about your files..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.chat_message("user", avatar="👨‍🎓").markdown(prompt)
            with st.chat_message("assistant", avatar="🤖"):
                template = """
                You are a helpful AI Tutor. Answer the user's question using the context provided below.
                If the context does not contain the answer, you MAY use your own general knowledge to answer.
                <context>{context}</context> Question: {input}
                """
                response = query_llm(llm, st.session_state.vector_store, prompt, template)
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

    # === TAB 2: QUIZ ===
    with tab2:
        c1, c2 = st.columns([3, 1])
        c1.markdown("#### 📝 Test your knowledge")
        if c2.button("🔄 Generate New Quiz"):
            st.session_state.quiz_data = []
            st.rerun()

        if not st.session_state.quiz_data:
            with st.status("🧠 Designing your quiz...", expanded=True) as status:
                # AGGRESSIVE PROMPT ENGINEERING
                quiz_prompt_template = """
                You are a strict professor. Create 5 multiple choice questions based on the provided text.

                STRICT RULES:
                1. DO NOT number the questions (e.g. NO "1.", NO "Q1:"). Start lines with the text directly.
                2. DO NOT use bolding or markdown in the question text.
                3. Options must be concise (max 15 words).

                FORMAT: Question Text Here ||| Option A ||| Option B ||| Option C ||| Option D ||| Correct Index (0-3) ||| Explanation

                <context>{context}</context>
                """
                all_questions = []
                st.write("Reading documents...")
                # Fetch more context (k=15) to avoid repetition
                raw_1 = query_llm(llm, st.session_state.vector_store, "Key concepts and definitions part 1",
                                  quiz_prompt_template)
                all_questions.extend(ParsingEngine.parse_quiz(raw_1))

                if len(all_questions) < 10:
                    st.write("Drafting more questions...")
                    time.sleep(1)  # FIXED: Sleep is now defined
                    raw_2 = query_llm(llm, st.session_state.vector_store, "Advanced applications and details part 2",
                                      quiz_prompt_template)
                    all_questions.extend(ParsingEngine.parse_quiz(raw_2))

                final_questions = all_questions[:10]
                if final_questions:
                    st.session_state.quiz_data = final_questions
                    status.update(label="✅ Quiz Ready!", state="complete", expanded=False)
                    st.rerun()
                else:
                    status.update(label="❌ Failed", state="error")
                    st.error("Failed to generate quiz. Try again.")

        if st.session_state.quiz_data:
            with st.form("quiz_form"):
                answers = {}
                for i, q in enumerate(st.session_state.quiz_data):
                    with st.container(border=True):
                        st.markdown(f"**{i + 1}. {q['question']}**")
                        answers[i] = st.radio("Select:", q['options'], key=f"q_{i}", index=None,
                                              label_visibility="collapsed")
                submitted = st.form_submit_button("Submit Answers", type="primary")
                if submitted:
                    score = 0
                    for i, q in enumerate(st.session_state.quiz_data):
                        correct_idx = q['correct']
                        if correct_idx < len(q['options']):
                            correct_choice = q['options'][correct_idx]
                        else:
                            correct_choice = "Error"
                        if answers[i] == correct_choice: score += 1

                    if score == len(st.session_state.quiz_data):
                        st.balloons()
                        st.success(f"🏆 Perfect Score! {score}/{len(st.session_state.quiz_data)}")
                    else:
                        st.warning(f"You got {score}/{len(st.session_state.quiz_data)}")
                        with st.expander("👀 Review Answers"):
                            for i, q in enumerate(st.session_state.quiz_data):
                                correct_idx = q['correct']
                                correct_txt = q['options'][correct_idx] if correct_idx < len(
                                    q['options']) else "Unknown"
                                if answers[i] != correct_txt:
                                    st.markdown(f"**Q{i + 1}:** {q['question']}")
                                    st.error(f"Your answer: {answers[i]}")
                                    st.success(f"Correct: {correct_txt}")
                                    st.info(f"💡 {q['explanation']}")
                                    st.divider()

    # === TAB 3: FLASHCARDS ===
    with tab3:
        c1, c2 = st.columns([3, 1])
        c1.markdown("#### 🧠 Active Recall")
        if c2.button("🔄 New Deck"):
            st.session_state.flashcards = []
            st.rerun()

        if not st.session_state.flashcards:
            with st.spinner("✍️ Writing flashcards..."):
                fc_prompt = """
                Create 5 flashcards based on the text.
                RULES:
                1. Front must be a SPECIFIC TERM or CONCEPT (not "Term", not "Concept").
                2. Back must be a detailed definition (2-3 sentences).
                3. Do not number them.

                FORMAT: Specific Term|||Detailed Definition
                <context>{context}</context>
                """
                all_cards = []
                raw_fc_1 = query_llm(llm, st.session_state.vector_store, "Key terms and definitions", fc_prompt)
                all_cards.extend(ParsingEngine.parse_flashcards(raw_fc_1))
                if len(all_cards) < 10:
                    raw_fc_2 = query_llm(llm, st.session_state.vector_store, "Advanced terminology", fc_prompt)
                    all_cards.extend(ParsingEngine.parse_flashcards(raw_fc_2))

                final_cards = all_cards[:10]
                if final_cards:
                    st.session_state.flashcards = final_cards
                    st.rerun()
                else:
                    st.error("⚠️ AI could not generate cards. Please try clicking 'New Deck' again.")

        if st.session_state.flashcards:
            total = len(st.session_state.flashcards)
            if st.session_state.fc_index >= total:
                st.session_state.fc_index = total - 1

            idx = st.session_state.fc_index
            card = st.session_state.flashcards[idx]

            st.progress((idx + 1) / total)
            st.caption(f"Card {idx + 1} of {total}")

            # Color Logic
            border_color = "#26C6DA" if st.session_state.fc_flipped else "#006C85"
            lbl = "DEFINITION" if st.session_state.fc_flipped else "TERM"
            content = card['back'] if st.session_state.fc_flipped else card['front']

            # --- INLINE CSS FORCED VISIBILITY ---
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
                <div style="
                    font-size: 24px; 
                    color: #000000 !important; /* FORCED BLACK */
                    font-weight: 600;
                    line-height: 1.5;">
                    {content}
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_prev, col_flip, col_next = st.columns([1, 2, 1])
            with col_prev:
                if st.button("⬅️ Previous", use_container_width=True):
                    if idx > 0:
                        st.session_state.fc_index -= 1
                        st.session_state.fc_flipped = False
                        st.rerun()
            with col_flip:
                if st.button("🔄 FLIP CARD", type="primary", use_container_width=True):
                    st.session_state.fc_flipped = not st.session_state.fc_flipped
                    st.rerun()
            with col_next:
                if st.button("Next ➡️", use_container_width=True):
                    if idx < total - 1:
                        st.session_state.fc_index += 1
                        st.session_state.fc_flipped = False
                        st.rerun()
                    else:
                        st.success("End of Deck!")