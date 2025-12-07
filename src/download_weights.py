from huggingface_hub import hf_hub_download
import shutil
import os

def setup_model():
    print("⬇️  Starting Model Download...")
    os.makedirs("models", exist_ok=True)
    
    # We will use Kerem Berke's YOLOv8m Table Extraction model
    # It is the most stable open-source model for this task.
    repo_id = "keremberke/yolov8m-table-extraction"
    filename = "best.pt" # The standard weight filename in HF
    
    try:
        # 1. Download from Hugging Face
        cached_path = hf_hub_download(repo_id=repo_id, filename=filename)
        print(f"   ✅ Downloaded to cache: {cached_path}")
        
        # 2. Move to our project folder
        final_path = "models/table_detector.pt"
        shutil.copy(cached_path, final_path)
        print(f"   ✅ Moved to: {final_path}")
        
        return final_path
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print("   Try running: pip install huggingface_hub")

if __name__ == "__main__":
    setup_model()