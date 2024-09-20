import subprocess
import os
from com.iisc.cds.cohort7.grp11 import rag_based_qna

def main():
    
    hfapi_key = "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" #getpass("Enter you HuggingFace access token:")
    os.environ["HF_TOKEN"] = hfapi_key
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hfapi_key

#fil_path = os.path.join(os.path.dirname(__file__), "nomura", "genai", "chat_ui.py")
    
    #print(f'File path: {fil_path}')
    
    #process = subprocess.Popen(["streamlit", "run", fil_path])
    os.environ["CONFIG_PATH"] = os.path.join(os.path.dirname(__file__), "../", "config.properties")
    
    print(f"config path: { os.getenv('CONFIG_PATH')}")
    
    embedding_model = "flax-sentence-embeddings/all_datasets_v3_MiniLM-L12"
    llm_model = "HuggingFaceH4/zephyr-7b-beta"
    #llm_model = "QuantFactory/finance-Llama3-8B-GGUF"
    
    query = "please suggest strategies to get regular income from my investments"
    rag_based_qna.generate_response(query, 1, llm_model, embedding_model)

if __name__ == "__main__":
    main()