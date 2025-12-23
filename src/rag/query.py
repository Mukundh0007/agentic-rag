import os
from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
    VectorStoreIndex,
)
try:
    from src.rag.openrouter_client import OpenRouterLLM, OpenRouterEmbedding
except ImportError:
    from openrouter_client import OpenRouterLLM, OpenRouterEmbedding

# Load env variables
load_dotenv()

# --- Configuration ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# LLM (Retriever & Synthesizer)
llm = OpenRouterLLM(
    model="openai/gpt-4o-mini",
    api_key=OPENROUTER_API_KEY,
)

# Embed Model (Must match ingest)
embed_model = OpenRouterEmbedding(
    model_name="openai/text-embedding-3-small", 
    api_key=OPENROUTER_API_KEY,
)

Settings.llm = llm
Settings.embed_model = embed_model

def query_system(user_query: str, persist_dir: str = "./storage") -> dict:
    """
    Takes a user query, retrieves relevant context (text + table summaries),
    and returns the answer along with source image paths.
    """
    
    # 1. Load the Index
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        return {"response": "Error: Storage not found. Run ingest.py first.", "images": []}

    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    
    # 2. Retrieve Context
    # We use the lower-level retriever to inspect nodes manually
    retriever = index.as_retriever(similarity_top_k=15)
    nodes = retriever.retrieve(user_query)
    
    # 3. Process Retrieved Nodes
    context_str = ""
    retrieved_images = []
    
    for node in nodes:
        # Extract Metadata
        page = node.metadata.get("page_label", "N/A")
        file_name = node.metadata.get("file_name", "N/A")
        
        # Accumulate text with clear headers for the LLM
        if "image_path" in node.metadata:
             context_str += f"\n--- Source: Table Image ({file_name}) ---\n{node.text}\n"
        else:
             context_str += f"\n--- Source: Text (Page {page}) ---\n{node.text}\n"
        
        # Check for image metadata
        if "image_path" in node.metadata:
            img_path = node.metadata["image_path"]
            if img_path not in retrieved_images:
                retrieved_images.append(img_path)
    
    # 4. Synthesize Answer
    system_prompt = (
        "You are a financial analyst assistant. "
        "Answer the user's question based ONLY on the context provided below. "
        "The context includes text from the report and summaries of data tables. "
        "Crucially, cite the Page Number (e.g., 'Page 22') or Table filename for every fact you state."
    )
    
    full_prompt = (
        f"{system_prompt}\n\n"
        f"Context:\n{context_str}\n\n"
        f"User Question: {user_query}\n"
        "Answer:"
    )
    
    response = llm.complete(full_prompt)
    
    return {
        "response_text": response.text,
        "source_images": retrieved_images,
        "context_used": context_str # Optional: for debug
    }

if __name__ == "__main__":
    # Test CLI
    q = "What was the total net sales in 2024?"
    print(f"❓ Query: {q}")
    result = query_system(q)
    print("\n💬 Response:")
    print(result["response_text"])
    print("\n🖼️  Source Images:")
    for img in result["source_images"]:
        print(f" - {img}")
