# - Researches and Studies AI assistant
import streamlit as st
import re
from sentence_transformers import SentenceTransformer
import plotly.express as px
from transformers import pipeline
from datetime import datetime
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader
import faiss
import os
import tempfile
import torch
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Researches/Studies AI Assistant",
                   page_icon="📑", layout="wide",
                   initial_sidebar_state="expanded")

# CSS for better styling
st.markdown("""
<style>
    .main_header {})
       font-size: 2.5rem;
       color: #1E3A8A;
       margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1.5rem;
    }
    .highlight {
        backgroud-color: #F0F9FF;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #3B82F6;
    }  
    .stProgress > div > div > div > div {
        background-color: #3B82F6;
    } 
    .stButton > button {
        width: 100%;
        margin-top: 5px;
    }
    .chat-message {
        padding: 1rem;
        border-radius:0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #f0f9ff;
        border-left: 4px solid #3B82F6;
    }
    .chat-message.assistant {
        background-color: #f7f7f7;
        border-left: 4px solid #10b981;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_models():
    """Load models once and cache them"""
    # better embedding model for academic text
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # FLAN-T5
    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=-1
    )
    return embedding_model, generator

def extract_text_from_pdf(pdf_file) -> Tuple[str, Dict[str, Any]]:
    """Extract text from PDF with metadata"""
    try:
        pdf_reader = PdfReader(pdf_file)
        # Extract text from all pages
        text= ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        # Extract metadata
        metadata = {
            "title": pdf_reader.metadata.get('/Title', 'Unknown') or 'Unknown',
            "author": pdf_reader.metadata.get('Author', 'Unknown') or 'Unknown',
            "subject": pdf_reader.metadata.get('/Subject', '') or '',
            "pages": len(pdf_reader.pages),
            "creation_date": pdf_reader.metadata.get('/CreationDate', '') or'',
            "file_size": len(pdf_file.getvalue()),
            "filename": pdf_file.name
        }
        return text, metadata
    except Exception as e:
        st.error(f"Error loading research: {str(e)}")
        return "", {}
def clean_chunk(chunk: str) -> str:
    """Remove headers, footers, page numbers, and other junk"""
    # Remove page numbers and standalone numbers
    chunk = re.sub(r'\b\d{1,4}\b', '', chunk)
    # remove all CAPS words
    chunk = re.sub(r'\b[A-Z]{3,}\b', '', chunk)
    # Remove short all CAPS seuqence (OF, IN, ON, TO)
    chunk = re.sub(r'b[A-Z]{2}\b', '', chunk)
    # Remove emails
    chunk = re.sub(r'\S+@\S+', '', chunk)
    # Remove URLs
    chunk = re.sub(r'https\S+|www\.\S+', '', chunk)
    # Remove excessive whitespace
    chunk = re.sub(r'\s+', ' ', chunk).strip()
    # Remove chunks that are too repetitive or have too many short words
    if len(chunk) < 30:
        return ""
    # Check if chunk has enough real content, not only prepositions and conjunctions
    words = chunk.split()
    content_words = [w for w in words if len(w) > 3]
    if len(content_words) < 3:
        return ""
    return chunk

def preprocess_research_text(text: str) -> List[str]:
    """Split the research paper into logival chunks"""
    text = re.sub(r'\s+', ' ', text)
    # try to split paper by sections
    section_patterns = [
        r'\n\s*(?:Abstract|INTRODUCTION|Introduction)\s*\n',
        r'\n\s*(?:METHODOLOGY|Methodology|METHODS|Methods)\s*\n',
        r'\n\s*(?:RESULTS|Results|FINDINGS|Findings)\s*\n',
        r'\n\s*(?:DISCUSSION|Discussion)\s*\n',
        r'\n\s*(?:CONCLUSION|Conclusion|CONCLUSIONS|Conclusios)\s*\n',
        r'\n\s*(?:REFERENCES|References|BIBLIOGRAPHY|Bibliography)\s*\n'
    ]
    chunks = []

    # try section-based splitting first
    for pattern in section_patterns:
        sections = re.split(pattern, text, flags=re.IGNORECASE)
        if len(sections) > 1:
            for section in sections:
                section = section.strip()
                if len(section) > 100: # the minimum chunk size
                    paragraphs = re.split(r'\.\s+|\n+', section)
                    for para in paragraphs:
                        para = para.strip()
                        if len(para) > 50: # the min paragraph length
                            chunks.append(para)
            break
    # if section based split is not working, then use sentence based chunking
    if len(chunks) < 5:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                current_chunk += sentence + " "
                if len(current_chunk) > 200: # the targeted chunk size
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
        if current_chunk:
            chunks.append(current_chunk.strip())
    # Clean the chunks
    chunks = [chunk for chunk in chunks if len(chunk) > 30]
    chunks = [clean_chunk(chunk) for chunk in chunks]
    chunks = [chunk for chunk in chunks if len(chunk) > 50]
    return chunks

def create_vector_store(chunks: List[str], embedding_model) -> faiss.Index:
    """Create FAISS index from text chunks"""
    if not chunks:
        raise ValueError("No text chunlks to index")
    # Generate embeddings
    embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    # Make FAISS index
    dimensions = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimensions)
    index.add(embeddings.astype('float32'))
    return index

def search_relevant_chunks(query: str, chunks: List[str], index: faiss.Index,
                           embedding_model, top_k: int = 5) -> List[Tuple[str, float]]:
    """Search for relevant chunks using semantic search"""
    if not chunks:
        return []
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    query_embedding = query_embedding.astype('float32')

    # search
    distances, indices = index.search(query_embedding, min(top_k, len(chunks)))

    # get chunks woth the scores (coverting distance to simialrity score)
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx < len(chunks):
            # convert L2 distance to similairyt score, the higher the better
            similarity = 1/ (1 + distance) if distance > 0 else 1.0
            results.append((chunks[idx], float(similarity)))
    return results

def generate_research_answer(question: str, relevant_chunks: List[Tuple[str, float]],
                            generator) -> str:
    """Generate answer based on research content"""
    if not relevant_chunks:
        return "I couldnt find relevant information in the papers"
    # Clean and prepare context - only use high-quality chunks
    context_parts = []
    for i, (chunk, score) in enumerate(relevant_chunks):
        # clean the chunk
        clean = clean_chunk(chunk)
        # Skip if chunk is too short or not meaningful after cleaning
        if len(clean) < 30:
            continue
        # Only use if relevance is decent
        if score > 0.4:
            context_parts.append(clean)
    if not context_parts:
        return "I found text but it is not clear enough to answer the question"
    # Use only the best chunks
    context = " ".join(context_parts[:2])
    # Truncate if  too long       
    if len(context) > 1000:
        context = context[:1000]
    # Better prompt with strict instructions
    prompt = f"""Based on the research text below, answer the question in 1-2 clear sentences.
Research text:
{context}
Question: {question}
Answer (1-2 sentences, no names or numbers):"""
    try:
        result = generator(
            prompt, max_length=100,
            min_length=15, temperature=0.3,
            do_sample=True, top_p=0.85, num_beams=4
        )
        answer = result[0]['generated_text'].strip()
        # Aggressive cleaning of the answer
        answer = answer.split("Answer:")[-1].strip()
        # remove standalooe capital words
        answer = re.sub(r'\b[A-Z]{2,}\b', '', answer)
        # remove excessive whitespace
        answer = re.sub(r'\s+', ' ', answer).strip()
        # if answer is still mostly junk, reject it
        words = answer.split()
        if len(words) < 5:
            return "I couldn't extract a clear answer from the available text"
        return answer
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def display_paper_metadata(metadata: Dict[str, Any]):
    """Display paper metadata in a good format"""
    with st.expander(f"📑 {metadata.get('filename', 'Paper Information')}", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            title = metadata.get('title', 'Unknown')
            if len(title) > 60:
                title = title[:60] + "..."
            st.metric("Title", title)
            author = metadata.get('author', 'unknown')
            if len(author) > 40:
                author = author[:40] + "..."
            st.metric("Author", author)
        with col2:
            st.metric("Pages", metadata.get('page', 0))
            file_size_kb = metadata.get('file_size', 0) / 1024
            st.metric("File size", f"{file_size_kb:.1f} KB")

def visualize_chunk_distribution(chunks: List[str]):
    """Visualize chunk length distribution"""
    if not chunks:
        return
    chunk_lengths = [len(chunk.split()) for chunk in chunks]
    if chunk_lengths:
        df = pd.DataFrame({'Word Count': chunk_lengths})

        fig = px.histogram(df, x='Word Count', title="Distribution of chunk lengths",
                           labels={'Word Count': 'Words per chunk'}, nbins=20,
                           color_discrete_sequence=['🟦 #3B82F6']
                           )
        fig.update_layout(showlegend=False, plot_bgcolor='white',
                          xaxis_title="Words per chunk", yaxis_title="Number of Chunks"
                          )
        # add average line
        avg_words = np.mean(chunk_lengths)
        fig.add_vline(x=avg_words,line_dash="dash",
                      line_color="red", annotation_text=f"Average: {avg_words:.1f} words",
                      annotation_position="top right"
                      )
        st.plotly_chart(fig, use_container_width=True)
# streamlit app
def main():
    st.markdown('<h1 class="main-header">🔬 Research paper AI assistant</h1>', unsafe_allow_html=True)
    st.markdown("Upload academic papers and ask questions about their content, methodology, findings, and conclusions")

    # Initialize models
    with st.spinner("Loading AI models (this may take a moment)..."):
        try:
            embedding_model, generator = initialize_models()
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"failed to load models: {str(e)}")
            st.info("pLEASE check your internet connection and try agian")
            return
    # sidebar
    with st.sidebar:
        st.header("Upload researches and studies")

        uploaded_files = st.file_uploader(
            "Choose PDF files",type=['pdf'],
            accept_multiple_files=True, help="Upload one or multiple research papers in PDF format"
        )
        st.markdown("---")
        st.header("Settings")
        top_k = st.slider(
            "Number of relevant passages to use",
            min_value=1, max_value=10, value=3,
            help="Higher vlaues use more context, might include less relevant information"
        )
        st.markdown("---")
        st.header("💡 example questions")

        # store example question in session state
        if "example_question" not in st.session_state:
            st.session_state.example_question = ""
        example_questions = ["What is the main research question?",
                             "What are the main findings in the paper"]
        for i, question in enumerate(example_questions):
            if st.button(f" {question}", key=f"example_{i}"):
                st.session_state.example_question = question
                st.rerun()

        st.markdown("---")
        if st.button("♻️ Clear chat history", use_container_width=True):
            if 'messages' in st.session_state:
                st.session_state.messages = []
            if 'exmaple_question' in st.session_state:
                st.session_state.example_question = ""
            st.rerun()
        st.markdown("---")
        st.caption("💡** Tip:** Upload multiple papers to ask comaprative questions")
    # main content area
    if uploaded_files:
        # Process uploaded files
        all_chunks = []
        all_metadata = []
        processing_successful = True

        progress_bar = st.progress(0, text="Processing PDF files..")
        for i, uploaded_file in enumerate(uploaded_files):
            progress_text = f"processing {uploaded_file.name}.."
            progress_bar.progress((i + 1) / len(uploaded_files), text=progress_text)

            with st.spinner(f"Reading {uploaded_file.name}.."):
                text, metadata = extract_text_from_pdf(uploaded_file)
                if text and len(text.strip()) > 50:
                    chunks = preprocess_research_text(text)
                    if chunks:
                        all_chunks.extend(chunks)
                        all_metadata.append({
                            'filename': uploaded_file.name,
                            **metadata, 'chunks': len(chunks)
                        })
                    else:
                        st.warning(f"Couldn't extract meaningful information from {uploaded_files.name}")
                        processing_successful = False
                else:
                    st.error(f"X failed to extract text from {uploaded_file.name}")
                    processing_successful = False
        progress_bar.empty()
        if all_chunks and processing_successful:
            # create vector store
            with st.spinner("creating the search index.."):
                try:
                    index = create_vector_store(all_chunks, embedding_model)

                    # store in session state for reuse
                    st.session_state.all_chunks = all_chunks
                    st.session_state.index = index
                    st.session_state.all_metadata = all_metadata

                    # display processing summary
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Papers", len(uploaded_files))
                    with col2:
                        st.metric("Text chunks", len(all_chunks))
                    with col3:
                        avg_chunks = len(all_chunks) // len(uploaded_files) if uploaded_files else 0
                    with col4:
                        st.metric("status", "Ready!")
                except Exception as e:
                    st.error(f"Failed to create a search index: {str(e)}")
                    return
            # display paper metadata
            with st.expander("📋 Paper details", expanded=False):
                for meta in all_metadata:
                    display_paper_metadata(meta)
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []
            # display chat history
            st.markdown("### 🗣️ Research Q&A")
            
            chat_container = st.container()

            with chat_container:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
            # Check if we have an example question to try
            question_to_process = None
            if st.session_state.example_question:
                question_to_process = st.session_state.example_question
                # Clear the example question after using it
                st.session_state.example_question = ""
            # Get question from the user
            question = question_to_process if question_to_process else st.chat_input("Ask about the research papers..")

            if question:
                # add user message to chat
                st.session_state.messages.append({"role": "user", "content": question})

                # Display user message
                with st.chat_message("user"):
                    st.markdown(question)
                # Generate response
                with st.chat_message("assisitant"):
                    with st.spinner("Searching paper and generating the answer.."):
                        # Search for relevant chunks
                        relevant_chunks = search_relevant_chunks(question, all_chunks, index,
                                                                 embedding_model, top_k)
                        if relevant_chunks:
                            # Generated answer
                            answer = generate_research_answer(question, relevant_chunks, generator)

                            # Display answer
                            st.markdown(answer)

                            # Show sources
                            with st.expander("📚 the source parts used"):
                                for i, (chunk, score) in enumerate(relevant_chunks):
                                    st.markdown(f"**Source {i+1}** (Relevance: {score:.3f})")
                                    st.info(chunk[:400] + ".." if len(chunk) > 400 else chunk)
                                    st.divider()
                        else:
                            st.warning("No reelvant passages found in the uploaded papers")
                            answer = "I couldnt find any relevant information in the papers to answer this question"
                # Add assistant response to history
                st.session_state.messages.append({
                    "role": "assistant", "content": answer,
                    "sources": [chunk for chunk, _ in relevant_chunks] if relevant_chunks else []
                })
                # Rerun and refresh the chat
                st.rerun()
        else:
            st.error("Could not extract sufficent text from the uploaded files, try different files")
            st.info("**Tips for better PDF processing:**")
            st.info("""
            1. Ensure PDFs are text based and not scanned images
            2. Check of PDFs have selectable text
            3. Try different files
            4. For scanned PDFs, and use OCR software firstly.
            """)
    else:
        # Welcome screen when no files are uploaded
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            ### 🎯 **Features:**
            ** 📑 Multi-Paper Analysis**
            - Upload multiple research papers
            - cross-paper searching and comaprison
            **🔎 Intelligent Q&A**
            - Ask about methodolgies, findings, conclusion
            - Get citations to specific passages
            - Understand complex academic content
            **📊 Analysis tools**
            _ Visualize paper structure
            _ Extract key information
            _ Compare across papers
                        
            ### 📝 ** Supported Question Types:**
            1. **Methodology Questions**: *What experimental design was used?"
            2. **Results Questions**: "What were the statistical findings?"
            3. **Comparative Questions**: "How do the methods differ between papers?"
            4. **Detail Questions**: "What sample size was used?"
            5. **Conceptual Questions**: "What theories are discussed?"
            """)   
        with col2:
            st.info("""
            ### 💡**Quick start:**
            1. Upload PDF papers in the sidebar
            2. Wait for processing (≈30 seconds per paper)
            3. Start asking questions                    
                    
            ### 📝** sample questions:**
            "Summarize the introduction"
            "Lists Limitations mentioned"
            "Discuss conclusions"
            """)
            
            st.warning("""
            ### ⚠️ **Requirements: **
            - PDFs must have selectable text
            - Internet connections required for first run
            - 4GB+ RAM recommended
            """)
        # Add sample papers option
        st.divider()
        st.markdown('### 🚀 Ready to start?')
        st.markdown("Upload your research papers with the uploader in the sidebar ->")
if __name__ == "__main__":
    main()