# 📘 Studee: AI-Powered Revision Companion

**Studee** is an intelligent revision platform that transforms static documents (PDF, DOCX, PPTX) into interactive study aids. Built as the Final Project for the Certified AI Engineer (CAIE) program.

🔗 **Live Demo:** [PASTE YOUR HUGGING FACE LINK HERE]

---

## 🚀 Features

*   **🤖 Multi-Modal RAG:** Uses Computer Vision (`LlamaParse`) to read and interpret complex tables, charts, and diagrams within documents.
*   **💬 Context-Aware Chat:** A strict RAG (Retrieval-Augmented Generation) chatbot that answers questions based *only* on your uploaded notes.
*   **📝 Exam Simulator:** Generates diverse, randomized multiple-choice quizzes with automatic scoring and detailed explanations.
*   **🧠 Smart Flashcards:** Automatically creates definitions and key concepts with a flip-card animation.
*   **🎯 Learning Outcomes:** Automatically extracts and summarizes the key takeaways from your files.
*   **📂 Multi-Format Support:** Accepts PDFs, Word Documents, and PowerPoint slides.

---

## 🛠️ Tech Stack

*   **Frontend:** Streamlit (Python)
*   **LLM:** Llama-3-8b (via Groq API) - *Fast inference*
*   **Computer Vision / OCR:** LlamaParse (LlamaIndex) - *Layout analysis*
*   **Vector Database:** FAISS - *Similarity search*
*   **Embeddings:** HuggingFace (all-MiniLM-L6-v2) - *Text vectorization*

---

## ⚙️ Setup & Installation

To run this project locally on your machine, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/bananavvv/studee.git
cd studee

2. Install Dependencies
Ensure you have Python 3.10 or 3.11 installed.
code
Bash
pip install -r requirements.txt

3. Configure API Keys
This app requires API keys to function. Create a file named .env in the root directory and add the following:
code
Env
GROQ_API_KEY=your_groq_api_key_here
LLAMA_CLOUD_API_KEY=your_llamacloud_api_key_here
(Get free keys from Groq Cloud and LlamaCloud)

4. Run the Application
code
Bash
streamlit run app.py