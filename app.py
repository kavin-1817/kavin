import streamlit as st
import time
import random
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
# ✅ Move to the first Streamlit command
st.set_page_config(page_title="DeepDocAI", page_icon="🤖", layout="wide")

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Custom CSS for better UI styling
st.markdown("""
    <style>
    .chat-bubble {
        background-color: #DCF8C6;
        color: black;
        padding: 12px;
        border-radius: 12px;
        max-width: 80%;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 8px;
        animation: fadeIn 0.5s ease-in-out;
    }
    .ai-bubble {
        background-color: #ECECEC;
        color: black;
        padding: 12px;
        border-radius: 12px;
        max-width: 80%;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 8px;
        animation: fadeIn 0.5s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .typing {
        font-size: 14px;
        color: #888;
        animation: blink 1.5s infinite;
    }
    @keyframes blink {
        0% { opacity: 0.2; }
        50% { opacity: 1; }
        100% { opacity: 0.2; }
    }
    </style>
""", unsafe_allow_html=True)

def get_pdf_text(pdf_docs):
    """Extract text from PDFs."""
    text = " "
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text

def get_text_chunks(text):
    """Split extracted text into smaller chunks for efficient processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Store text embeddings in FAISS vector storage."""
    if not text_chunks:
        st.error("❌ No text found in the uploaded PDFs. Please check the document.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        batch_size = 10  # Process embeddings in smaller batches
        for i in range(0, len(text_chunks), batch_size):
            chunk_batch = text_chunks[i : i + batch_size]
            vector_store = FAISS.from_texts(chunk_batch, embedding=embeddings)
            vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"⚠️ Error in vector storage: {e}")

def get_conversational_chain():
    """Set up the AI conversational chain."""
    prompt_template = """
You are an advanced AI assistant with strong reasoning abilities. Your task is to provide **clear, well-structured, and fact-based answers** strictly using the given context.

### **Guidelines:**  
1. If the answer **exists in the context**, provide a **concise, direct, and well-explained response**.  
2. If the answer **is not found directly**, analyze the context carefully:  
   - If the context provides indirect hints, use reasoning to give the most accurate answer possible.  
   - If the context explicitly states the **opposite** of what is being asked, state that clearly.  
   - If the answer is **completely unavailable**, respond in a human-like way by saying:  
     **"No, [subject] does not [action]."**  
     or  
     **"There is no information available in the provided context to determine this."**  
3. **Use bullet points and line breaks for clarity when listing multiple points.**  
4. **Never make up information that is not in the context.**  

### **Context:**  
{context}  

### **User Question:**  
{question}  

### **Answer:**  
"""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def format_response(response_text):
    """Format response for Streamlit Markdown (with bullet points on new lines)."""
    formatted_text = response_text.replace("-", "\n-")  # ✅ Ensure new lines for each bullet point
    formatted_text = formatted_text.replace("\n", "\n\n")  # ✅ Ensure proper spacing between lines
    return formatted_text

def display_animated_text(response_text):
    """Display AI response with a typing effect inside a chat bubble."""
    placeholder = st.empty()
    displayed_text = ""

    with placeholder.container():
        st.markdown('<div class="typing">🤖 AI is typing...</div>', unsafe_allow_html=True)
        time.sleep(1.5)  # Simulate AI thinking time

    formatted_text = format_response(response_text)  # Apply formatting

    for char in formatted_text:
        displayed_text += char
        placeholder.markdown(f'<div class="ai-bubble">{displayed_text}</div>', unsafe_allow_html=True)
        time.sleep(random.uniform(0.01, 0.04))  # Random delay for realistic typing effect

def user_input(user_question):
    """Retrieve relevant documents and generate AI response with animation."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    # Display user question in chat UI
    st.markdown(f'<div class="chat-bubble">🧑‍💼 **You:** {user_question}</div>', unsafe_allow_html=True)

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    display_animated_text(response["output_text"])

def main():
    st.header("🚀 DeepDocAI - Chat with PDFs")

    user_question = st.text_input("🔍 Ask a Question from the Uploaded Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📂 Upload PDF Documents:")
        pdf_docs = st.file_uploader("Upload your PDFs and Click on Submit", accept_multiple_files=True)
        
        if st.button("📥 Submit & Process"):
            with st.spinner("⚡ Processing PDFs... Please wait."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("✅ Processing Complete! You can now ask questions.")

if __name__ == "__main__":
    main()
