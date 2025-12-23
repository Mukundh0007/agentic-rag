import streamlit as st
import sys
import os

# Ensure src module is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from rag.query import query_system
except ImportError:
    # Fallback if running from src directory or other context
    from src.rag.query import query_system

# --- Page Configuration ---
st.set_page_config(
    page_title="Agentic RAG - Financial Analyst",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Agentic RAG: Visual Financial Assistant")


# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Render History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If there are images associated with this message (assistant only)
        if "images" in message and message["images"]:
            with st.expander("🔍 Verified Source Tables", expanded=True):
                cols = st.columns(len(message["images"]))
                for idx, img_path in enumerate(message["images"]):
                    if os.path.exists(img_path):
                        st.image(img_path, caption=f"Source: {os.path.basename(img_path)}")
                    else:
                        st.warning(f"Image not found: {img_path}")

# --- Handle User Input ---
if prompt := st.chat_input("Ask a question about the financial report (e.g., 'What was the revenue in 2024?')..."):
    
    # 1. Display User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Save to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing document and table images..."):
            try:
                # Call the RAG pipeline
                result = query_system(prompt)
                
                response_text = result.get("response_text", "No response generated.")
                source_images = result.get("source_images", [])

                # Display Text
                st.markdown(response_text)
                
                # Display Images (The "Wow" Factor)
                if source_images:
                    with st.expander("🔍 Verified Source Tables", expanded=True):
                        # Use columns for better layout if multiple images
                        num_cols = min(3, len(source_images))
                        if num_cols > 0:
                            cols = st.columns(num_cols)
                            for idx, img_path in enumerate(source_images):
                                col = cols[idx % num_cols]
                                if os.path.exists(img_path):
                                    col.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
                                else:
                                    col.warning("Source image file missing.")

                # Save Response and Images to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_text,
                    "images": source_images
                })
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
