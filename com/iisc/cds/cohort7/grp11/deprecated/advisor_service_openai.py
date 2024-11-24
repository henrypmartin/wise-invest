'''
Created on 03-Sep-2024

@author: Henry Martin
'''
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from com.iisc.cds.cohort7.grp11 import config_reader

import os

instance_llm = None
instance_retriever = None

class OpenAIAdvisorService:
    
    def __init__(self):
        config_reader.load_config()            
        self.llm_model = "gpt-4o"        
        self.index_path = config_reader.get_property('local', 'financial_data_index')
        
        os.environ["OPENAI_API_KEY"] = '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'
        
        print(f'LLM Model: {self.llm_model}')
        print(f'Index Path: {self.index_path}')

    class Config:
        arbitrary_types_allowed = True
            
    def qna_llm(self):
        llm = ChatOpenAI(model=self.llm_model, temperature=0.1)
        return llm
    
    def rag_retriever(self):
        #embedding = OpenAIEmbeddings()
        #vectorstore = InMemoryVectorStore(embedding)
        #vectorstore = vectorstore.load(index_path, embedding)
        
        model_name = "flax-sentence-embeddings/all_datasets_v3_MiniLM-L12"
        embedding = HuggingFaceEndpointEmbeddings(model=model_name)
        
        combined_vectorstore = None
        all_files = os.listdir(self.index_path)
        
        if all_files.count('combined.faiss') == 1:
            combined_vectorstore = FAISS.load_local(self.index_path, embedding, index_name='combined', allow_dangerous_deserialization=True)
        else:
            for file_name in all_files:
                if file_name.endswith('daily_stocks_data_part58.faiss'):
                    print(f'Loading index {file_name}')
                    vectorstore = FAISS.load_local(self.index_path, embedding, index_name=file_name[:-6], allow_dangerous_deserialization=True)
                    
                    if not combined_vectorstore:
                        combined_vectorstore = vectorstore
                    else:
                        combined_vectorstore.merge_from(vectorstore)
            
            #combined_vectorstore.save_local(index_path, index_name='combined')
        
        return combined_vectorstore.as_retriever()

    def rag_retriever_orig(self):
        embedding = OpenAIEmbeddings()
        #vectorstore = InMemoryVectorStore(embedding)
        #vectorstore = vectorstore.load(index_path, embedding)
        
        vectorstore = FAISS.load_local(self.index_path, embedding, allow_dangerous_deserialization=True)
        
        return vectorstore.as_retriever()

    