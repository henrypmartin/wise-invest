'''
Created on 03-Sep-2024

@author: Henry Martin
'''
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores.faiss import FAISS

from com.iisc.cds.cohort7.grp11 import config_reader
from com.iisc.cds.cohort7.grp11.advisor_service import contextualize_q_prompt, qa_prompt, get_response

import os

instance_llm = None
instance_retriever = None

class Config:
    arbitrary_types_allowed = True
        
def qna_llm(model_id):
    llm = ChatOpenAI(model=model_id, temperature=0.1)
    #print(dir(llm))
    return llm

def rag_retriever(index_path):
    embedding = OpenAIEmbeddings()
    #vectorstore = InMemoryVectorStore(embedding)
    #vectorstore = vectorstore.load(index_path, embedding)
    
    combined_vectorstore = None
    all_files = os.listdir(index_path)
    
    if all_files.count('combined.faiss') == 1:
        combined_vectorstore = FAISS.load_local(index_path, embedding, index_name='combined', allow_dangerous_deserialization=True)
    else:
        for file_name in all_files:
            if file_name.endswith('.faiss'):
                print(f'Loading index {file_name}')
                vectorstore = FAISS.load_local(index_path, embedding, index_name=file_name[:-6], allow_dangerous_deserialization=True)
                
                if not combined_vectorstore:
                    combined_vectorstore = vectorstore
                else:
                    combined_vectorstore.merge_from(vectorstore)
        
        combined_vectorstore.save_local(index_path, index_name='combined')
    
    return combined_vectorstore.as_retriever()

def rag_retriever_orig(index_path):
    embedding = OpenAIEmbeddings()
    #vectorstore = InMemoryVectorStore(embedding)
    #vectorstore = vectorstore.load(index_path, embedding)
    
    vectorstore = FAISS.load_local(index_path, embedding, allow_dangerous_deserialization=True)
    
    return vectorstore.as_retriever()

def generate_response_for_model(query, session_id, llm_model_id, index_path):    
    
    print(f"Index file path: {index_path}")
    
    global instance_llm
    global instance_retriever
    
    if not instance_llm:
        instance_llm = qna_llm(llm_model_id)
    
    if not instance_retriever:
        instance_retriever = rag_retriever(index_path)

    #chat_index_retriever = rag_retriever(index_path)

    print('retriever loaded')
    
    history_aware_retriever = create_history_aware_retriever(instance_llm, instance_retriever, contextualize_q_prompt)

    question_answer_chain = create_stuff_documents_chain(instance_llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    #print(type(rag_chain))
    #rag_chain.invoke(input, config)
    response = get_response(rag_chain, query, session_id)
        
    print(f'AI Answer: {response}')        
       
    return response

def generate_response(query, session_id):
    
    os.environ["OPENAI_API_KEY"] = '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'
    
    llm_model = "gpt-4o"
    
    config_reader.load_config()
    
    index_path = config_reader.get_property('local', 'financial_data_index')    
    #index_path = os.path.join(index_path, "embeddings.index")
    
    findata = generate_response_for_model(query, session_id, llm_model, index_path)
    
    index_path = config_reader.get_property('local', 'personal_finance_index')
    index_path = os.path.join(index_path, "embeddings.index")
    
    #personal_findata = generate_response_for_model(query, session_id, llm_model, index_path)
    
    #data = "Output using finance data:\n"
    data = findata
    #data += '\n****************************************************\n'
    #data += "Output using personal finance data:\n"
    #data += personal_findata
    
    return data
    
    