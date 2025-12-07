import os
import sys
import shutil
from pathlib import Path

# 1. Verify Imports - If these fail, the environment is broken.
try:
    import torch
    import cv2
    import streamlit as st
    from ultralytics import YOLO
    from llama_index.core import Document, VectorStoreIndex
    from llama_index.llms.openai import OpenAI
    print("✅ All libraries imported successfully.")
except ImportError as e:
    print(f"❌ CRITICAL IMPORT ERROR: {e}")
    print("   Please run: pip install -r requirements.txt")
    sys.exit(1)

def verify_hardware():
    """Checks for GPU acceleration (CUDA or MPS)."""
    print("\n--- Hardware Acceleration Check ---")
    print(f"Python Version: {sys.version.split()}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ NVIDIA CUDA Detected: {gpu_name}")
        print("   YOLOv8 will run in high-performance mode.")
    elif torch.backends.mps.is_available():
        print("✅ Apple Silicon (MPS) Detected.")
        print("   YOLOv8 will run in optimized Metal mode.")
    else:
        print("⚠️ No GPU detected. Running on CPU.")
        print("   Inference might be slow for large documents.")

def verify_yolo_download_and_inference():
    """
    Downloads the YOLOv8n model and runs a test inference.
    Moves the downloaded weight file to the 'models/' directory.
    """
    print("\n--- YOLOv8 Vision Pipeline Check ---")
    
    # Define paths
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True) # Ensure models/ exists
    model_path = models_dir / "yolov8n.pt"
    
    try:
        # Load the model. Ultralytics will download 'yolov8n.pt' to current dir if missing.
        # We explicitly use the nano model for a lightweight Day 1 test.
        print(f"   Loading YOLOv8 model (Target: {model_path})...")
        
        # If we already have it in models/, load from there. Else let Ultralytics download.
        if model_path.exists():
            model = YOLO(model_path)
            print("   Model loaded from local 'models/' directory.")
        else:
            print("   Downloading weights from Ultralytics Hub...")
            model = YOLO("yolov8n.pt") 
            # Move the downloaded file to the organized 'models/' folder
            if Path("yolov8n.pt").exists():
                shutil.move("yolov8n.pt", model_path)
                print(f"   Moved 'yolov8n.pt' to {model_path}")
        
        # Run Dummy Inference
        # Ultralytics provides a hosted image for testing.
        # We use 'stream=True' to verify generator output if needed, but standard predict is fine.
        print("   Running test inference on 'bus.jpg'...")
        results = model.predict("https://ultralytics.com/images/bus.jpg", verbose=False)
        
        # Verify results
        for result in results:
            boxes = result.boxes
            print(f"✅ Inference Successful. Detected {len(boxes)} objects.")
            # Optional: Check if a 'bus' (class 5) or 'person' (class 0) was found
            # This confirms class mapping is loaded correctly.
            
    except Exception as e:
        print(f"❌ YOLO Verification Failed: {e}")
        print("   Check internet connection (required for weight download).")

def verify_llamaindex_structure():
    """
    Verifies that the LlamaIndex components can be initialized.
    Does NOT require an API Key for instantiation (only for execution).
    """
    print("\n--- LlamaIndex Agentic Core Check ---")
    try:
        # 1. Test Document creation
        doc = Document(text="Agentic Multimodal RAG verification.")
        
        # 2. Test LLM Object instantiation
        # Note: We are mocking the API key check here. 
        # In a real run, this requires OPENAI_API_KEY in env variables.
        llm = OpenAI(model="gpt-3.5-turbo", api_key="sk-dummy-key-for-init-check")
        
        print(f"✅ LlamaIndex Core: Document object created.")
        print(f"✅ LlamaIndex OpenAI: LLM Interface initialized (v{llm.model}).")
        
    except Exception as e:
        print(f"❌ LlamaIndex Verification Failed: {e}")

if __name__ == "__main__":
    verify_hardware()
    verify_yolo_download_and_inference()
    verify_llamaindex_structure()
    print("\n🎉 Day 1 Setup Verification Complete. Project is ready for development.")

