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
    
model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEndpointEmbeddings(model=model_name)

transformer = SentenceTransformer(model_name)

model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
#embeddings = HuggingFaceEmbeddings(model_name=model_name,
#                                   model_kwargs=model_kwargs,
#                                   encode_kwargs=encode_kwargs)

print('loaded embeddings model')
docs = []
src_file = "C:\\Henry\\Learning\\IICS_CDS_Capstone\\scrapped_data\\financial_data\\back_up\\daily_stocks_data_part30.txt"
with open(src_file, 'r', encoding='utf-8') as file:    
    file_text = file.read()
    #doc = Document(page_content=file_text, metadata={"name": "daily_stocks_data_part30.txt"})
    
    docs.append(file_text)

print('loaded file')
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 150)

#all_splits = text_splitter.split_documents(docs)
all_splits = text_splitter.split_text(docs[0])
print('text split')

data_embeddings = transformer.encode(all_splits)
print('embeddings done')

vector_store = FAISS.from_embeddings(data_embeddings, embeddings)
print('vector store created')
#data_index = VectorstoreIndexCreator(text_splitter=text_splitter, embedding=embeddings, vectorstore_cls=FAISS)

print('creating index')
#db_index = data_index.from_documents(docs)

print('saving index')
#db_index.vectorstore.save_local("C:\\Henry\\Learning\\IICS_CDS_Capstone\\scrapped_data\\financial_data", index_name="daily_stocks_data_part30")
vector_store.save_local("C:\\Henry\\Learning\\IICS_CDS_Capstone\\scrapped_data\\financial_data", index_name="daily_stocks_data_part30")

query='what is the latest stock price of reliance industries and as of what date?'
vec_docs = vector_store.similarity_search(query, k=5)

print(vec_docs)
