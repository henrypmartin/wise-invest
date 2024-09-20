import subprocess
import os
from com.iisc.cds.cohort7.grp11 import data_loader

def main():
    
    hfapi_key = "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" #getpass("Enter you HuggingFace access token:")
    os.environ["HF_TOKEN"] = hfapi_key
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hfapi_key

#fil_path = os.path.join(os.path.dirname(__file__), "nomura", "genai", "chat_ui.py")
    
    #print(f'File path: {fil_path}')
    
    #process = subprocess.Popen(["streamlit", "run", fil_path])
    os.environ["CONFIG_PATH"] = os.path.join(os.path.dirname(__file__), "../", "config.properties")
    
    print(f"config path: { os.getenv('CONFIG_PATH')}")
    data_loader.process_data("flax-sentence-embeddings/all_datasets_v3_MiniLM-L12")
    
    #"jinaai/jina-embeddings-v2-base-zh"

if __name__ == "__main__":
    main()