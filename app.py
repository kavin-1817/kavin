
import streamlit as st
import time
import random
import os
import tempfile
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pdf2image import convert_from_path
import pytesseract
from concurrent.futures import ThreadPoolExecutor

# ✅ Tesseract path configuration for Windows
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\tesseract\tesseract.exe'

st.set_page_config(page_title="DeepDocAI", page_icon="🤖", layout="wide")

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Custom CSS remains the same...

def process_pdf(pdf):
    """Process a single PDF with proper temp file handling"""
    text = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf.getbuffer())
            temp_path = temp_file.name

        # Extract text from PDF pages
        pdf_reader = PdfReader(temp_path)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"

        # Extract text from images using OCR (Tamil language)
        images = convert_from_path(temp_path)
        for image in images:
            # Set Tesseract language to Tamil (tam)
            text += pytesseract.image_to_string(image, lang='tam') + "\n"

    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return text

def get_pdf_text(pdf_docs):
    """Improved parallel processing with error handling"""
    text = ""
    if not pdf_docs:
        return text

    progress_bar = st.progress(0)
    total_pdfs = len(pdf_docs)

    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_pdf, pdf) for pdf in pdf_docs]
            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    text += result + "\n"
                except Exception as e:
                    st.error(f"Failed to process a PDF: {str(e)}")
                progress_bar.progress((i + 1) / total_pdfs)
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
    
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
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
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
        time.sleep(0.5)  # Reduce the thinking time

    formatted_text = format_response(response_text)

    for char in formatted_text:
        displayed_text += char
        placeholder.markdown(f'<div class="ai-bubble">{displayed_text}</div>', unsafe_allow_html=True)
        time.sleep(random.uniform(0.005, 0.02))  # Reduce the typing delay

def user_input(user_question):
    """Retrieve relevant documents and generate AI response with animation."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    # Display user question in chat UI
    st.markdown(f'<div class="chat-bubble">🧑‍💼 You: {user_question}</div>', unsafe_allow_html=True)

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    display_animated_text(response["output_text"])

def main():
    st.header("🚀 DeepDocAI - Chat with PDFs (Tamil Support)")
    
    # Add system requirement note
   

    user_question = st.text_input("🔍 Ask a Question from the Uploaded Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📂 Upload PDF Documents:")
        pdf_docs = st.file_uploader("Upload your PDFs and Click on Submit", accept_multiple_files=True)
        
        if st.button("📥 Submit & Process"):
            if not pdf_docs:
                st.warning("⚠️ Please upload at least one PDF file")
                return
                
            with st.spinner("⚡ Processing PDFs... Please wait."):
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text.strip():
                    st.error("❌ Failed to extract text from PDFs. Check if documents contain text/images.")
                    return
                
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("✅ Processing Complete! You can now ask questions.")

if __name__ == "__main__":
    main()
