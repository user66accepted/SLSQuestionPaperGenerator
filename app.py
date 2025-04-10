import streamlit as st
import pdfplumber
import os
import uuid
import re
import glob
import hashlib
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq

# Set up persistent directory for ChromaDB
PERSIST_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

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

# Get list of existing PDF collections
def get_existing_pdfs():
    # Map to store collection name -> display name
    pdf_collections = {}
    
    # Look for collection directories in the ChromaDB persist directory
    chroma_collections = glob.glob(os.path.join(PERSIST_DIRECTORY, "*"))
    for coll_path in chroma_collections:
        if os.path.isdir(coll_path):
            coll_name = os.path.basename(coll_path)
            # Extract original PDF name from collection name
            if coll_name.startswith("pdf_"):
                pdf_id = coll_name[4:]  # Remove 'pdf_' prefix
                # Store with a user-friendly display name
                pdf_collections[pdf_id] = pdf_id
    
    return pdf_collections

# Sanitize filename to be used as collection name
def sanitize_filename(filename):
    # Remove file extension
    base_name = os.path.splitext(filename)[0]
    
    # Create a hash of the original name to ensure uniqueness
    name_hash = hashlib.md5(base_name.encode()).hexdigest()[:8]
    
    # Sanitize the name, keep only alphanumeric, underscores and hyphens
    sanitized = re.sub(r'[^\w\-]', '_', base_name)
    
    # Trim to ensure it's not too long (max 50 chars + 8 char hash = 58 chars)
    if len(sanitized) > 50:
        sanitized = sanitized[:50]
    
    # Ensure it starts and ends with alphanumeric
    if not sanitized[0].isalnum():
        sanitized = 'p' + sanitized[1:]
    if not sanitized[-1].isalnum():
        sanitized = sanitized[:-1] + 'z'
    
    # Add hash to ensure uniqueness while keeping name recognizable
    sanitized = f"{sanitized}_{name_hash}"
    
    return sanitized

# Cache PDF loading to avoid reprocessing
def load_pdf(file) -> str:
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Cache text splitting
def split_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)

# Store chunks in vector database
def store_in_vectordb(chunks, pdf_name):
    embeddings = get_embeddings_model()
    
    # Create a new collection for this PDF using its name
    collection_name = f"pdf_{pdf_name}"
    db = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    
    # Add documents to the collection
    db.add_texts(chunks)
    db.persist()
    
    return collection_name

# Check if PDF exists in the database
def pdf_exists_in_db(pdf_name):
    existing_pdfs = get_existing_pdfs()
    return pdf_name in existing_pdfs

# RAG retrieval function
def retrieve_relevant_chunks(query, collection_name, top_k=5):
    embeddings = get_embeddings_model()
    
    # Load the collection for this PDF
    db = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    
    # Retrieve the most relevant chunks
    results = db.similarity_search(query, k=top_k)
    return [doc.page_content for doc in results]

# Generate questions using RAG and Groq cloud API
def generate_questions(
    collection_name,
    num_questions,
    difficulty,
    conceptuality,
    model_name="llama3-70b-8192"
) -> str:
    # Create a query to retrieve relevant content for question generation
    retrieval_query = f"Generate {difficulty} {conceptuality} questions about the following content"
    
    # Retrieve the most relevant chunks for question generation
    relevant_chunks = retrieve_relevant_chunks(retrieval_query, collection_name, top_k=3)
    
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
    
    # Initialize session state for selected PDF
    if "selected_collection" not in st.session_state:
        st.session_state["selected_collection"] = None
    
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
    
    # Get existing PDFs
    existing_pdfs = get_existing_pdfs()
    
    # Create tabs for uploading new PDF or selecting existing one
    tab1, tab2 = st.tabs(["Upload New PDF", "Use Existing PDF"])
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Upload PDF of Course Material", type="pdf"
        )

        if uploaded_file:
            # Get PDF name and sanitize it
            pdf_name = sanitize_filename(uploaded_file.name)
            
            # Check if this PDF has already been processed
            if pdf_exists_in_db(pdf_name):
                st.warning(f"A PDF with name '{uploaded_file.name}' already exists in the database.")
                if st.button("Use Existing Data"):
                    st.session_state["selected_collection"] = f"pdf_{pdf_name}"
                    st.success(f"Using existing data for '{uploaded_file.name}'")
                    
                if st.button("Reprocess PDF"):
                    process_new_pdf(uploaded_file, pdf_name)
            else:
                if st.button("Process PDF & Create Vector Database"):
                    process_new_pdf(uploaded_file, pdf_name)
    
    with tab2:
        if existing_pdfs:
            selected_pdf_id = st.selectbox(
                "Select Previously Processed PDF",
                options=list(existing_pdfs.keys()),
                format_func=lambda x: existing_pdfs[x]  # Display the friendly name
            )
            
            if st.button("Use Selected PDF"):
                collection_name = f"pdf_{selected_pdf_id}"
                st.session_state["selected_collection"] = collection_name
                st.success(f"Using '{existing_pdfs[selected_pdf_id]}' for question generation")
        else:
            st.info("No previously processed PDFs found. Please upload a PDF first.")
    
    # Show question generation section if a collection is selected
    if st.session_state["selected_collection"]:
        st.markdown("---")
        st.markdown("### Generate Questions")
        st.write(f"Using PDF: {st.session_state['selected_collection'][4:]}")  # Remove 'pdf_' prefix
        
        if st.button("Generate Questions"):
            collection_name = st.session_state["selected_collection"]
            
            with st.spinner("Retrieving relevant content and generating questions with Groq..."):
                result = generate_questions(
                    collection_name, num_questions, difficulty, conceptuality, model_name
                )
            
            st.markdown("### Generated Questions")
            st.write(result)

def process_new_pdf(uploaded_file, pdf_name):
    with st.spinner("Extracting text from PDF..."):
        text = load_pdf(uploaded_file)
    
    with st.spinner("Splitting text into chunks..."):
        chunks = split_text(text)
        st.success(f"Processed PDF into {len(chunks)} chunks.")
    
    with st.spinner("Creating vector embeddings and storing in database..."):
        collection_name = store_in_vectordb(chunks, pdf_name)
        st.success("Vector database created successfully!")
    
    # Store the collection name in session state
    st.session_state["selected_collection"] = collection_name

if __name__ == "__main__":
    main()
