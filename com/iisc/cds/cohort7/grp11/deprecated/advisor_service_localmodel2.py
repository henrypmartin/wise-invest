'''
Created on 09-Nov-2024

@author: Henry Martin
'''

from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from sentence_transformers import SentenceTransformer

import os

hfapi_key = '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' #getpass("Enter you HuggingFace access token:")
os.environ["HF_TOKEN"] = hfapi_key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hfapi_key
    
model_name = "flax-sentence-embeddings/all_datasets_v3_MiniLM-L12"
embeddings = HuggingFaceEndpointEmbeddings(model=model_name)

index_path="C:\\Henry\\Learning\\IICS_CDS_Capstone\\index\\flax-sentence-embeddings\\financial_data_index"
vector_store = FAISS.load_local(index_path, embeddings, index_name='daily_stocks_data_part10', allow_dangerous_deserialization=True)
print('vector store loaded')

query='what is the stock price of reliance industries as of April 2016?'
#vec_docs = vector_store.similarity_search(query, k=5)

vec_docs = vector_store.as_retriever().invoke(query)

print(vec_docs)
