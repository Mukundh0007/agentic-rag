import os
import glob
from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Document,
)
from llama_index.core.node_parser import SentenceSplitter
try:
    from src.rag.openrouter_client import OpenRouterLLM, OpenRouterEmbedding
except ImportError:
    from openrouter_client import OpenRouterLLM, OpenRouterEmbedding
from llama_index.core.schema import TextNode

# Load environment variables
load_dotenv()

# --- Configuration ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Initialize LLM (OpenRouter)
# Models: "google/gemini-flash-1.5", "openai/gpt-4o", etc.
llm = OpenRouterLLM(
    model="openai/gpt-4o-mini",
    api_key=OPENROUTER_API_KEY,
)

# Initialize Embeddings (OpenRouter)
embed_model = OpenRouterEmbedding(
    model_name="openai/text-embedding-3-small", 
    api_key=OPENROUTER_API_KEY,
)

# Set Global Settings
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 1024
Settings.chunk_overlap = 20

def summarize_table_image(image_path: str) -> str:
    """
    Sends table image to VLM to get a text summary.
    """
    import base64
    
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        
    # Construct Multimodal Message for OpenRouter/OpenAI-compatible
    # Note: LlamaIndex OpenAI class supports passing `image_url` in messages
    
    from llama_index.core.llms import ChatMessage, MessageRole, ImageBlock, TextBlock

    # Create content blocks
    image_block = ImageBlock(
        url=f"data:image/png;base64,{base64_image}",
        detail="high"  # Optional, for OpenAI
    )
    
    prompt_text = (
        "Analyze this image of a financial table. "
        "Output a comprehensive text summary of the data it contains, "
        "including column headers and key row values, so that it can be retrieved via search. "
        "Do not include Markdown formatting like ```json or ```text, just the clean summary."
    )
    
    text_block = TextBlock(text=prompt_text)
    
    # Send request
    try:
        response = llm.chat(
            messages=[
                ChatMessage(
                    role=MessageRole.USER,
                    blocks=[text_block, image_block]
                )
            ]
        )
        return response.message.content
    except Exception as e:
        print(f"Error summarising {image_path}: {e}")
        return f"Error processing table: {os.path.basename(image_path)}"

def build_pipeline(pdf_path, table_output_dir, persist_dir="./storage"):
    """
    Args:
        pdf_path (str): Path to the uploaded PDF.
        table_output_dir (str): Directory where extracted table images are located.
        persist_dir (str): Directory to save the vector index.
    """
    print(f"🚀 Starting RAG Ingestion Pipeline for: {pdf_path}")

    # 1. Load & Chunk PDF Text
    # ---------------------------
    from llama_index.readers.file import PDFReader
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found at {pdf_path}")
        return None

    print(f"📄 Loading Text from PDF...")
    
    # Force PDF parsing
    parser = PDFReader()
    file_extractor = {".pdf": parser}
    reader = SimpleDirectoryReader(input_files=[pdf_path], file_extractor=file_extractor)
    
    pdf_docs = reader.load_data()
    
    # Create Text Nodes (Chunks)
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    text_nodes = splitter.get_nodes_from_documents(pdf_docs)
    print(f"✅ Generated {len(text_nodes)} text nodes from PDF.")

    # 2. Multimodal Table Processing (Parallelized)
    # ---------------------------
    image_files = glob.glob(os.path.join(table_output_dir, "*.png"))
    table_nodes = []
    
    if image_files:
        print(f"🖼️  Found {len(image_files)} table images. Starting Parallel VLM processing...")
        
        import concurrent.futures

        # Helper for parallel execution
        def process_image(img_path):
            try:
                summary = summarize_table_image(img_path)
                return img_path, summary
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                return img_path, None

        # Run with ThreadPool
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_img = {executor.submit(process_image, img): img for img in image_files}
            
            for i, future in enumerate(concurrent.futures.as_completed(future_to_img)):
                img_path, summary = future.result()
                print(f"   [{i+1}/{len(image_files)}] Analyzed {os.path.basename(img_path)}", end="\r")
                
                if summary:
                    node = TextNode(text=summary)
                    node.metadata = {
                        "image_path": img_path,
                        "file_name": os.path.basename(img_path),
                        "type": "table_image",
                        "page_num": "unknown" 
                    }
                    table_nodes.append(node)

        print(f"\n✅ Generated {len(table_nodes)} table nodes from images.")
    else:
        print("ℹ️  No table images found to process.")

    # 3. Embed Everything & 4. Persist
    # ---------------------------
    all_nodes = text_nodes + table_nodes
    print(f"🧠 Embedding {len(all_nodes)} total nodes ({len(text_nodes)} text + {len(table_nodes)} tables)...")
    
    # Create Index
    index = VectorStoreIndex(
        nodes=all_nodes, 
        show_progress=True
    )
    
    # Save to Disk
    index.storage_context.persist(persist_dir=persist_dir)
    print(f"💾 Index persisted to {persist_dir}")
    print("🎉 Pipeline Finish!")
    return index

if __name__ == "__main__":
    # Default for CLI compatibility
    build_pipeline(
        pdf_path=os.path.abspath("data/apple_10k.pdf"),
        table_output_dir=os.path.abspath("data/processed_tables"),
        persist_dir="./storage"
    )
