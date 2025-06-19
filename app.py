import os 
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st
from dotenv import load_dotenv
import time

load_dotenv()

#groq_api_key = os.environ.get("groq_api_key")


if "vectors" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    st.session_state.loader = PyPDFLoader('C:\\Users\\PRADEESHKUMARG\\Desktop\\projects\\rag_with_opensource_llms\\bda unit 3.pdf')
    st.session_state.doc = st.session_state.loader.load()
    
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap =100)
    st.session_state.final_doc = st.session_state.text_splitter.split_documents(st.session_state.doc)
    st.session_state.vectors = chroma.Chroma.from_documents(
        st.session_state.final_doc,st.session_state.embeddings)


st.title("question-answers generator with groq")

llm = ChatGroq(groq_api_key = "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",model_name = "gemma2-9b-it")


prompt = ChatPromptTemplate.from_template(
    """You are an intelligent tutor and content generator.

Given the following:
- A document: {context}
- A topic of focus: "{topic}"
- A difficulty level: "{difficulty_level}" (choose from Easy, Medium, or High)
- A question type: "{question_type}" (choose from "MCQs", "Short Answer Questions", or "Long Answer Questions")

Your task is to:
1. Understand the document and extract key information related to the topic.
2. Generate 5 high-quality questions based on the topic and document content.
3. Ensure the questions match the requested difficulty level and question type.
4. Provide a clear and accurate answer for each question.

### Output Format:

**Question 1:** [Question here]  
**Answer:** [Answer here]

**Question 2:** [Question here]  
**Answer:** [Answer here]

...and so on up to 5 questions.

Make the questions educational, relevant to the topic, and aligned with the specified difficulty and type.

"""
)


document_chain = create_stuff_documents_chain(llm,prompt)
retriver = st.session_state.vectors.as_retriever()
retrival_chain = create_retrieval_chain(retriver,document_chain)


topic = st.text_input("Enter the topic (e.g., Neural Networks)")
difficulty = st.selectbox("Select difficulty level", ["Easy", "Medium", "High"])
question_type = st.selectbox("Select question type", ["MCQs", "Short Answer Questions", "Long Answer Questions"])


start = time.process_time()
response = retrival_chain.invoke({
    "input": topic,
    "topic": topic,
    "difficulty_level": difficulty,
    "question_type": question_type
})
print("Response time:",round(time.process_time() - start),"seconds.")
st.write(response['answer'])

