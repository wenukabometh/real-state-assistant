# 🔎 Web Q&A Assistant using Groq + LangChain + Streamlit

This project is a lightweight RAG (Retrieval-Augmented Generation) app that answers questions based on the content of any webpage you provide.

It uses:

- 🧠 **LLaMA3-70B** via [Groq API](https://console.groq.com/)
- 🧬 **HuggingFace Sentence Transformers** for embeddings
- 🗂️ **ChromaDB** for fast in-memory vector search
- 🌐 **LangChain** to orchestrate the RAG pipeline
- 🎨 **Streamlit** for a beautiful UI

---

## 🚀 Features

✅ Ask questions from **any website**  
✅ Fast, accurate answers via **Groq's LLaMA3-70B**  
✅ Clean UI with **Streamlit**  
✅ Fully open source & extendable

---

## 📸 Demo

![Initial UI](./assets/initial_ui.png)
![Initial UI](./assets/ui_after_generating_answers.png)

---

## 🔧 Setup

### 1. Clone the Repo

```bash
git clone https://github.com/wenukabometh/real-state-assistant.git
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Your Groq API Key

```bash
GROQ_API_KEY=your_groq_api_key_here

```

### ▶️ Run the App

```bash
streamlit run app.py

```

### 📝 How to Use

Paste one or more webpage URLs (comma-separated).

Ask your question in the input box.

Click "Ask".

Get a precise answer, with source URLs.

### 🙏 Acknowledgement

This project was completed as part of the Gen AI & Data Science on Codebasics.
