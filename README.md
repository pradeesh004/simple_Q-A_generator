# simple_Q-A_generator
# 📘 Question-Answer Generator using Groq & Ollama

This is a **Streamlit-based web application** that generates **questions and answers** from a PDF document using:

- 💬 **Groq LLMs** (e.g., `gemma2-9b-it`)
- 📎 **Ollama embeddings** for vector search
- 🧠 **LangChain** for retrieval-augmented generation (RAG)

---

## 🔧 Features

- Loads a local PDF file and splits it into chunks
- Lets the user input a **topic**, **difficulty**, and **question type**
- Uses a Groq language model to generate 5 questions and answers
- Displays the results inside the app

---

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- Groq API
- Ollama embeddings
- Chroma for vector storage

---

## 🚀 How to Run

1. Clone the repository or copy the script.
2. Create and activate a virtual environment (optional but recommended).
3. Install required packages:

```bash
pip install -r requirements.txt

Create a .env file and add your Groq API key:

env
Copy
Edit
groq_api_key=your_groq_api_key_here
Run the Streamlit app:

bash
Copy
Edit
streamlit run app.py
