import subprocess
import os
from com.iisc.cds.cohort7.grp11 import data_indexer
from getpass import getpass

def main():
    
    hfapi_key = getpass("Enter you HuggingFace access token:")
    os.environ["HF_TOKEN"] = hfapi_key
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hfapi_key

    os.environ["CONFIG_PATH"] = os.path.join(os.path.dirname(__file__), "../", "config.properties")
    
    print(f"config path: { os.getenv('CONFIG_PATH')}")
    data_indexer.process_data("flax-sentence-embeddings/all_datasets_v3_MiniLM-L12", 'webscrapped')
    
    #"jinaai/jina-embeddings-v2-base-zh"

if __name__ == "__main__":
    main()