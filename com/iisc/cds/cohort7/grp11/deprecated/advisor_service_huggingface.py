'''
Created on 03-Sep-2024

@author: Henry Martin
'''
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores.faiss import FAISS
from com.iisc.cds.cohort7.grp11 import config_reader
from langchain_openai import  OpenAIEmbeddings

import os

class HuggingFaceAdvisorService:    
    
    def __init__(self):
        config_reader.load_config()            
        self.llm_model = "HuggingFaceH4/zephyr-7b-beta"        
        self.embedding_model = "flax-sentence-embeddings/all_datasets_v4_MiniLM-L6"        
        self.index_path = config_reader.get_property('local', 'financial_data_index')
        
        hfapi_key = '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' #getpass("Enter you HuggingFace access token:")
        os.environ["HF_TOKEN"] = hfapi_key
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hfapi_key
        
        print(f'LLM Model: {self.llm_model}')
        print(f'Embedding Model: {self.embedding_model}')
        print(f'Index Path: {self.index_path}')
        
    def qna_llm(self):
        llm = HuggingFaceEndpoint(                
            repo_id=self.llm_model,
            #task="text-generation",
            task="question-answering",
            max_new_tokens = 512,
            top_k = 20,
            temperature = 0.1,
            repetition_penalty = 1.03,
            )
        return llm

    def rag_retriever_openai(self):
        embedding = OpenAIEmbeddings()
        #vectorstore = InMemoryVectorStore(embedding)
        #vectorstore = vectorstore.load(index_path, embedding)
        
        combined_vectorstore = None
        all_files = os.listdir(self.index_path)
        
        if all_files.count('combined.faiss') == 1:
            combined_vectorstore = FAISS.load_local(self.index_path, embedding, index_name='combined', allow_dangerous_deserialization=True)
        else:
            for file_name in all_files:
                if file_name.endswith('.faiss'):
                    print(f'Loading index {file_name}')
                    vectorstore = FAISS.load_local(self.index_path, embedding, index_name=file_name[:-6], allow_dangerous_deserialization=True)
                    
                    if not combined_vectorstore:
                        combined_vectorstore = vectorstore
                    else:
                        combined_vectorstore.merge_from(vectorstore)
            
            combined_vectorstore.save_local(self.index_path, index_name='combined')
        
        return combined_vectorstore.as_retriever()
    
    def rag_retriever(self):
        embedding = HuggingFaceEndpointEmbeddings(model=self.embedding_model)
        
        all_files = os.listdir(self.index_path)
        combined_vectorstore = None
        
        for file_name in all_files:
            if file_name.endswith('.faiss'):            
                print(f'Loading index {file_name}')
                vectorstore = FAISS.load_local(self.index_path, embedding, index_name=file_name[:-6], allow_dangerous_deserialization=True)
                
                if not combined_vectorstore:
                    combined_vectorstore = vectorstore
                else:
                    combined_vectorstore.merge_from(vectorstore)
                        
        #index = FAISS.load_local(self.index_path, embedding, index_name='daily_stocks_data_part58', allow_dangerous_deserialization=True)
        
        return combined_vectorstore.as_retriever()
    