import streamlit as st
import pdfplumber
import os
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from groq import Groq
import easyocr
from PIL import Image
import io
import sys
import numpy as np

# Hardcoded Groq API Key
GROQ_API_KEY = "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"

# Initialize embedding model
@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Groq client
@st.cache_resource
def get_groq_client():
    return Groq(api_key=GROQ_API_KEY)

# Initialize EasyOCR reader
@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(['en'])

# Load PDF and extract text
def load_pdf(file) -> str:
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Extract text from image using OCR
def extract_text_from_image(image_file) -> str:
    # Get the OCR reader
    reader = get_ocr_reader()
    
    # Open image and convert to numpy array
    image = Image.open(image_file)
    image_np = np.array(image)
    
    # Perform OCR
    results = reader.readtext(image_np)
    
    # Extract text from results
    text = "\n".join([result[1] for result in results])
    return text

# Process multiple image files
def process_images(image_files) -> str:
    combined_text = ""
    progress_bar = st.progress(0)
    
    for i, image_file in enumerate(image_files):
        with st.spinner(f"Processing image {i+1}/{len(image_files)}..."):
            text = extract_text_from_image(image_file)
            combined_text += text + "\n\n"
        
        # Update progress
        progress = (i + 1) / len(image_files)
        progress_bar.progress(progress)
    
    return combined_text

# Cache text splitting
def split_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)

# Simple search function using embedding similarity
def retrieve_relevant_chunks(query, chunks, top_k=5):
    embeddings = get_embeddings_model()
    
    # Get the query embedding
    query_embedding = embeddings.embed_query(query)
    
    # Calculate similarities between query and all chunks
    chunk_embeddings = embeddings.embed_documents(chunks)
    
    # Calculate similarities using dot product
    similarities = []
    for i, chunk_emb in enumerate(chunk_embeddings):
        # Simple dot product similarity
        similarity = sum(q_e * c_e for q_e, c_e in zip(query_embedding, chunk_emb))
        similarities.append((similarity, i))
    
    # Sort by similarity (highest first) and get top_k indices
    top_indices = [idx for _, idx in sorted(similarities, reverse=True)[:top_k]]
    
    # Return the top chunks
    return [chunks[idx] for idx in top_indices]

# Generate questions using Groq cloud API
def generate_questions(
    chunks,
    num_questions,
    difficulty,
    conceptuality,
    model_name="llama3-70b-8192"
) -> str:
    # Create a query to retrieve relevant content for question generation
    retrieval_query = f"Generate {difficulty} {conceptuality} questions about the following content"
    
    # Retrieve the most relevant chunks for question generation
    relevant_chunks = retrieve_relevant_chunks(retrieval_query, chunks, top_k=3)
    
    # Build the prompt for question generation
    prompt = (
        f"You are a helpful teaching assistant creating a question paper. Based on the following content, generate exactly {num_questions} distinct, well-formed questions. "
        f"The difficulty should be {difficulty}, and conceptuality level should be {conceptuality}.\n\n"
        f"Format your response as a numbered list of questions (1., 2., 3., etc.). "
        f"Do not write as 'mentioned in text' or 'from text' or 'from the text'etc when generating questions"
        f"Each question should be detailed, clear, and directly related to the material. "
        f"Do not generate MCQ questions. "
        f"Do not provide answers or explanations in your response, only the questions themselves.\n\n"
        f"Content:\n{' '.join(relevant_chunks)}"
    )
    
    # Initialize Groq client
    client = get_groq_client()
    
    # Generate questions using Groq API
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful teaching assistant that generates clear educational questions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=4000
    )
    
    return completion.choices[0].message.content

# Streamlit App
def main():
    st.title("SLS Question Paper Generator")
    
    # Initialize session state for storing chunks
    if "chunks" not in st.session_state:
        st.session_state["chunks"] = None
    if "source_name" not in st.session_state:
        st.session_state["source_name"] = None
    
    st.sidebar.header("Parameters")
    difficulty = st.sidebar.selectbox(
        "Difficulty", ["Easy", "Medium", "Hard"]
    )
    conceptuality = st.sidebar.selectbox(
        "Conceptuality Level", ["Low", "Medium", "High"]
    )
    num_questions = st.sidebar.number_input(
        "Number of Questions", min_value=1, max_value=50, value=10
    )
    
    # Model selection
    model_options = {
        "LLaMA 3 70B": "llama3-70b-8192",
        "LLaMA 3 8B": "llama3-8b-8192",
        "Mixtral 8x7B": "mixtral-8x7b-32768",
        "Gemma 7B": "gemma-7b-it"
    }
    selected_model = st.sidebar.selectbox("Select Groq Model", list(model_options.keys()))
    model_name = model_options[selected_model]
    
    # Choose input type (PDF or Images)
    input_type = st.radio("Select Input Type", ["PDF Document", "Book Images"])
    
    if input_type == "PDF Document":
        # Upload PDF section
        uploaded_file = st.file_uploader(
            "Upload PDF of Course Material", type="pdf"
        )

        if uploaded_file:
            if st.button("Process PDF"):
                with st.spinner("Extracting text from PDF..."):
                    text = load_pdf(uploaded_file)
                
                with st.spinner("Splitting text into chunks..."):
                    chunks = split_text(text)
                    st.session_state["chunks"] = chunks
                    st.session_state["source_name"] = uploaded_file.name
                    st.success(f"Processed '{uploaded_file.name}' into {len(chunks)} chunks.")
    
    else:  # Images option
        st.info("The first time you use image processing, it may take a minute to download the OCR model.")
        
        # Upload multiple images
        uploaded_images = st.file_uploader(
            "Upload Images of Book Pages", type=["jpg", "jpeg", "png"], accept_multiple_files=True
        )
        
        if uploaded_images:
            if st.button("Process Images"):
                try:
                    st.info(f"Processing {len(uploaded_images)} images with EasyOCR...")
                    
                    with st.spinner("Extracting text from images using OCR..."):
                        text = process_images(uploaded_images)
                        
                    # Display extracted text for review
                    with st.expander("Review extracted text"):
                        st.text_area("Extracted text", text, height=200, disabled=True)
                    
                    with st.spinner("Splitting text into chunks..."):
                        chunks = split_text(text)
                        st.session_state["chunks"] = chunks
                        st.session_state["source_name"] = f"{len(uploaded_images)} book images"
                        st.success(f"Processed {len(uploaded_images)} images into {len(chunks)} chunks.")
                except Exception as e:
                    st.error(f"Error processing images: {str(e)}")
    
    # Show question generation section if chunks are available
    if st.session_state["chunks"]:
        st.markdown("---")
        st.markdown("### Generate Questions")
        st.write(f"Using: {st.session_state['source_name']}")
        
        if st.button("Generate Questions"):
            chunks = st.session_state["chunks"]
            
            with st.spinner("Generating questions with Groq..."):
                result = generate_questions(
                    chunks, num_questions, difficulty, conceptuality, model_name
                )
            
            st.markdown("### Generated Questions")
            st.write(result)

if __name__ == "__main__":
    main()
