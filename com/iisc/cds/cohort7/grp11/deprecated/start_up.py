import subprocess
import os
from com.iisc.cds.cohort7.grp11.deprecated import data_indexer
from getpass import getpass

#hfapi_key = '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' #getpass("Enter you HuggingFace access token:")
#os.environ["HF_TOKEN"] = hfapi_key
#os.environ["HUGGINGFACEHUB_API_TOKEN"] = hfapi_key

def main():
    
    
    os.environ["CONFIG_PATH"] = os.path.join(os.path.dirname(__file__), "../", "config.properties")
    
    print(f"config path: { os.getenv('CONFIG_PATH')}")
    #data_indexer.process_data("flax-sentence-embeddings/all_datasets_v4_MiniLM-L6", 'financedata')
    #data_indexer.process_data("amazon/Titan-text-embeddings-v2", 'webscrapped')
    data_indexer.process_data(None, 'financedata')
    
    #"jinaai/jina-embeddings-v2-base-zh"

if __name__ == "__main__":
    main()