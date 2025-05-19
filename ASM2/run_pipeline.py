import os
import sys
import time
import argparse
from subprocess import run, PIPE

def run_stage(script_path, description):
    """Run a stage of the pipeline and print the output."""
    print(f"\n{'=' * 80}")
    print(f"STAGE: {description}")
    print(f"{'=' * 80}")
    
    # Run the script and capture output
    result = run([sys.executable, script_path], stdout=PIPE, stderr=PIPE, text=True)
    
    # Print stdout and stderr
    if result.stdout:
        print("\nOutput:")
        print(result.stdout)
    
    if result.stderr:
        print("\nErrors/Warnings:")
        print(result.stderr)
    
    # Check if the script completed successfully
    if result.returncode != 0:
        print(f"\n❌ Stage failed with return code {result.returncode}")
        return False
    
    print(f"\n✅ Stage completed successfully")
    return True

def create_directories():
    """Create necessary directories for the pipeline."""
    directories = [
        "processed_data",
        "chunked_data",
        "annotated_data",
        "vector_store"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def main():
    """Run the entire RAG pipeline."""
    parser = argparse.ArgumentParser(description="Run the RAG pipeline")
    parser.add_argument("--skip-data-cleaning", action="store_true", help="Skip data cleaning stage")
    parser.add_argument("--skip-chunking", action="store_true", help="Skip text chunking stage")
    parser.add_argument("--skip-annotation", action="store_true", help="Skip annotation stage")
    parser.add_argument("--skip-embedding", action="store_true", help="Skip embedding and vector store stage")
    parser.add_argument("--run-app-only", action="store_true", help="Only run the RAG application")
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("\n🚀 Starting RAG Pipeline")
    
    # Create directories
    create_directories()
    
    stages = []
    
    # Define pipeline stages
    if not args.run_app_only:
        if not args.skip_data_cleaning:
            stages.append(("data_cleaning.py", "Data Cleaning"))
        
        if not args.skip_chunking:
            stages.append(("text_chunker.py", "Text Chunking"))
        
        if not args.skip_annotation:
            stages.append(("annotator.py", "Annotation"))
        
        if not args.skip_embedding:
            stages.append(("embedding_engine.py", "Embedding and Vector Store Creation"))
    
    # Always add the RAG application stage
    stages.append(("rag_app.py", "RAG Application"))
    
    # Run each stage
    for script_path, description in stages:
        success = run_stage(script_path, description)
        
        if not success:
            print("\n❌ Pipeline failed at stage: " + description)
            return
    
    # Calculate and print total run time
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n🎉 Pipeline completed successfully!")
    print(f"Total run time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")

if __name__ == "__main__":
    main() 