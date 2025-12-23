import streamlit as st
import os
import sys
import shutil
import time

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from rag.query import query_system
    from rag.ingest import build_pipeline
    from vision.vision_processor import VisionProcessor
    from llama_index.core import StorageContext, load_index_from_storage
except ImportError:
    # Fallback
    from src.rag.query import query_system
    from src.rag.ingest import build_pipeline
    from src.vision.vision_processor import VisionProcessor
    from llama_index.core import StorageContext, load_index_from_storage

# --- Configuration ---
st.set_page_config(
    page_title="Visual Fin-Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)



# --- simple Auth ---
def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        if st.session_state["password"] == "admin": 
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.header("🔒 Login")
        st.text_input("Username", value="admin", disabled=True) # Demo convenience
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.info("Demo Password: 'admin'")
        return False
    
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input again
        st.header("🔒 Login")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct
        return True

if not check_password():
    st.stop()  # Stop execution if not logged in

# --- Main App Logic ---

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False
if "current_file" not in st.session_state:
    st.session_state.current_file = "None"

# Custom CSS for Layout Only
st.markdown("""
<style>
    /* Main Area Styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar: File Upload & Status ---
with st.sidebar:
    st.title("📂 Documents")
    
    uploaded_file = st.file_uploader("Upload Financial PDF (10-K)", type=["pdf"])
    
    if uploaded_file:
        file_name = uploaded_file.name
        
        # Check if this file is new
        if st.session_state.current_file != file_name:
            st.info(f"Ready to process: {file_name}")
            
            if st.button("✨ Process Document", type="primary"):
                
                # 1. Save File
                os.makedirs("data/uploads", exist_ok=True)
                pdf_path = os.path.join("data/uploads", file_name)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # 2. Vision Pipeline
                progress_bar = st.progress(0, text="Initializing Vision AI...")
                
                table_output_dir = os.path.join("data/uploads/tables", file_name.split('.')[0])
                
                try:
                    # Step A: Table Extraction
                    progress_bar.progress(10, text="Detecting Tables (YOLOv8)...")
                    vision = VisionProcessor(output_dir=table_output_dir)
                    extracted_images = vision.process_pdf(pdf_path)
                    
                    st.success(f"👁️ Extracted {len(extracted_images)} tables")
                    
                    # Step B: Ingestion & Indexing
                    progress_bar.progress(40, text="Analyzing Content (GPT-4o Vision)...")
                    
                    # Create dedicated storage for this file
                    persist_dir = f"./storage/{file_name.split('.')[0]}"
                    
                    # Store PDF path for viewer
                    st.session_state.current_pdf_path = pdf_path
                    
                    build_pipeline(
                        pdf_path=pdf_path,
                        table_output_dir=table_output_dir,
                        persist_dir=persist_dir
                    )
                    
                    progress_bar.progress(100, text="Ready!")
                    time.sleep(1)
                    progress_bar.empty()
                    
                    # Update State
                    st.session_state.current_file = file_name
                    st.session_state.index_ready = True
                    st.session_state.persist_dir = persist_dir 
                    
                    # Clear chat for new doc
                    st.session_state.messages = []
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Processing Failed: {e}")
                    st.progress(0).empty()

    if st.session_state.index_ready:
        st.success("✅ Index Active")
        st.caption(f"File: {st.session_state.current_file}")
    else:
        st.info("Please upload and process a document to begin.")
        
    st.divider()
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- Main Area: Split Screen Interface ---

if not st.session_state.index_ready:
    st.title("Visual RAG Financial Analyst")
    st.caption("Multimodal Financial Intelligence • Powered by Vision AI")
    st.markdown("""
    <div style='text-align: center; padding: 50px; color: #86868b;'>
        <h2>👋 Welcome</h2>
        <p>Upload a PDF document in the sidebar to get started.</p>
        <p>I can read text and <b>see tables</b> to answer your questions.</p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Creating a 2-column layout: PDF Viewer | Chat Interface
    col1, col2 = st.columns([1.2, 1]) 
    
    with col1:
        st.subheader("📄 Document Viewer")
        # Display PDF
        pdf_path = st.session_state.get("current_pdf_path", "data/apple_10k.pdf")
        if os.path.exists(pdf_path):
            from streamlit_pdf_viewer import pdf_viewer
            # Streamlit Cloud needs binary reading
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            pdf_viewer(input=pdf_bytes, width=700)
        else:
            st.warning("PDF file not found.")

    with col2:
        st.subheader("💬 AI Assistant")
        
        # Chat Container
        chat_container = st.container(height=700)
        
        with chat_container:
            # Display Chat History
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "images" in message and message["images"]:
                        with st.expander("🔍 Verified Source Tables", expanded=False):
                            cols = st.columns(min(3, len(message["images"])))
                            for idx, img_path in enumerate(message["images"]):
                                col = cols[idx % len(cols)]
                                if os.path.exists(img_path):
                                    col.image(img_path, caption=os.path.basename(img_path))

        # User Input (Bottom of Column)
        if prompt := st.chat_input("Ask a question about the document..."):
            
            # Add User Message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Add Assistant Message
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing..."):
                        try:
                            target_storage = st.session_state.get("persist_dir", "./storage")
                            result = query_system(prompt, persist_dir=target_storage) 
                            
                            response_text = result.get("response_text", "No response.")
                            source_images = result.get("source_images", [])
                            
                            st.markdown(response_text)
                            
                            if source_images:
                                with st.expander("🔍 Verified Source Tables", expanded=True):
                                    cols = st.columns(min(3, len(source_images)))
                                    for idx, img_path in enumerate(source_images):
                                        col = cols[idx % len(cols)]
                                        if os.path.exists(img_path):
                                            col.image(img_path, caption=os.path.basename(img_path))
                                            
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response_text,
                                "images": source_images
                            })
                            
                        except Exception as e:
                            st.error(f"Error: {e}")

