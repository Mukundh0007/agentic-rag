import argparse
import sys
import os
import subprocess

# Ensure src in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

def run_ingest():
    """Import and run the ingestion pipeline."""
    print("🚀 Triggering Ingestion Pipeline...")
    try:
        from rag.ingest import build_pipeline
        build_pipeline()
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        print("Tip: Check if your .env file is set up correctly.")

def run_app():
    """Launch the Streamlit web application."""
    print("🌐 Launching Streamlit App...")
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    try:
        subprocess.run(["streamlit", "run", app_path], check=True)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user.")
    except Exception as e:
        print(f"❌ Failed to launch app: {e}")

def run_query(question):
    """Run a single query from the command line."""
    print(f"❓ Querying: '{question}'")
    
    # Check for storage existence strictly
    if not os.path.exists("storage") or not os.listdir("storage"):
        print("❌ Database not found. Please run `python main.py --ingest` first.")
        return

    try:
        from rag.query import query_system
        result = query_system(question)
        
        print("\n💬 Response:")
        print("-" * 50)
        print(result["response_text"])
        print("-" * 50)
        
        if result["source_images"]:
            print(f"\n🖼️  Source Images Found ({len(result['source_images'])}):")
            for img in result["source_images"]:
                print(f"   - {img}")
        else:
            print("\nℹ️  No visual tables cited for this answer.")
            
    except Exception as e:
        print(f"❌ Query failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Agentic RAG Controller")
    
    # Create mutually exclusive group for commands
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ingest", action="store_true", help="Run the data ingestion pipeline (PDF + Tables -> Vector DB)")
    group.add_argument("--app", action="store_true", help="Launch the Streamlit web interface")
    group.add_argument("--query", type=str, help="Run a specific question in the terminal", metavar="\"QUESTION\"")

    args = parser.parse_args()

    if args.ingest:
        run_ingest()
    elif args.app:
        run_app()
    elif args.query:
        run_query(args.query)

if __name__ == "__main__":
    main()
