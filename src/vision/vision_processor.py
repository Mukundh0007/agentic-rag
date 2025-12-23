import fitz  # PyMuPDF
from ultralytics import YOLO
from PIL import Image
import os
import shutil

class VisionProcessor:
    def __init__(self, model_path="models/table_detector.pt", output_dir="data/processed_tables"):
        # Verify model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model not found at {model_path}. Run src/download_weights.py first!")
            
        print(f"👁️  Loading Vision Model: {model_path}...")
        self.model = YOLO(model_path)
        
        # Output setup
        self.output_dir = output_dir
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir) # Cleanup old runs
        os.makedirs(self.output_dir, exist_ok=True)

    def process_pdf(self, pdf_path):
        """Main pipeline: PDF Page -> Image -> YOLO Detect -> Crop Table"""
        if not os.path.exists(pdf_path):
            print(f"❌ PDF not found: {pdf_path}")
            return []

        doc = fitz.open(pdf_path)
        print(f"📄 Processing {len(doc)} pages from {pdf_path}...")
        
        tables_found = 0
        extracted_tables = []
        
        # Loop through pages
        for page_num, page in enumerate(doc):
            # 1. Render page to high-res image (300 DPI equivalent)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) 
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # 2. Run YOLO Inference
            results = self.model.predict(img, conf=0.25, verbose=False)
            
            # 3. Process Detections
            for result in results:
                for box in result.boxes:
                    coords = box.xyxy.cpu().tolist()
                    x1, y1, x2, y2 = map(int, coords[0])
                    
                    # Crop the table from the page
                    table_crop = img.crop((x1, y1, x2, y2))
                    
                    # Save locally
                    filename = f"p{page_num+1}_table_{tables_found}.png"
                    save_path = os.path.join(self.output_dir, filename)
                    table_crop.save(save_path)
                    
                    # print(f"   📸 Found Table on Page {page_num+1} -> Saved: {filename}")
                    extracted_tables.append(save_path)
                    tables_found += 1

        print(f"\n✅ Done! Extracted {tables_found} tables to '{self.output_dir}'")
        return extracted_tables

if __name__ == "__main__":
    # Test run
    pdf_path = "data/apple_10k.pdf" 
    try:
        processor = VisionProcessor()
        processor.process_pdf(pdf_path)
    except Exception as e:
        print(f"Error: {e}")