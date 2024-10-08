import os
from com.iisc.cds.cohort7.grp11 import advisor_service
from getpass import getpass
import subprocess

def main():
    
    hfapi_key = getpass("Enter you HuggingFace access token:")
    os.environ["HF_TOKEN"] = hfapi_key
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hfapi_key

    os.environ["CONFIG_PATH"] = os.path.join(os.path.dirname(__file__), "../", "config.properties")
    
    print(f"config path: { os.getenv('CONFIG_PATH')}")
    
    embedding_model = "nvidia/NV-Embed-v2"
    #llm_model = "HuggingFaceH4/zephyr-7b-beta" #working model
    llm_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    query = "What should be my retirement courpus to live comfortably with expected monthly income of 100000 rupees after retirement from y investments"
    #advisor_service.generate_response(query, 1, llm_model, embedding_model)
    
    file_path = os.path.join(os.path.dirname(__file__), "iisc", "cds", "cohort7", "grp11", "ui", "chat_ui.py")
    
    print(f'File path: {file_path}')
    
    process = subprocess.Popen(["streamlit", "run", file_path])

if __name__ == "__main__":
    main()