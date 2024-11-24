'''
Created on 03-Sep-2024

@author: Henry Martin
'''
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers import util
from com.iisc.cds.cohort7.grp11 import config_reader
import traceback
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from com.iisc.cds.cohort7.grp11.deprecated.advisor_service import contextualize_q_prompt, qa_prompt, get_response

import os

embedding_model_id = None
instance_llm = None
instance_retriever = None

class HuggingFaceAdvisorService:    
    
    def __init__(self):
        config_reader.load_config()            
        self.llm_model = "HuggingFaceH4/zephyr-7b-beta"        
        self.embedding_model = "flax-sentence-embeddings/all_datasets_v4_MiniLM-L6"        
        self.index_path = config_reader.get_property('local', 'financial_data_index')
        
        hfapi_key = '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' #getpass("Enter you HuggingFace access token:")
        os.environ["HF_TOKEN"] = hfapi_key
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hfapi_key
        
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
    
    def rag_retriever_hf(self):
        print(f'Model to use {embedding_model_id}')
        model_name = "flax-sentence-embeddings/all_datasets_v3_MiniLM-L12"
        embedding = HuggingFaceEndpointEmbeddings(model=model_name)
        
        index = FAISS.load_local(self.index_path, embedding, index_name='daily_stocks_data_part58', allow_dangerous_deserialization=True)
        
        return index.as_retriever()

    store = {}
    
    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:    
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    def validate_answer_against_sources(self, response_answer, source_documents):
        #model = SentenceTransformer('all-mpnet-base-v2')
        model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v3_MiniLM-L12')
            
        similarity_threshold = 0.7  
        source_texts = [doc.page_content for doc in source_documents]
        answer_embedding = model.encode(response_answer, convert_to_tensor=True)
        source_embeddings = model.encode(source_texts, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(answer_embedding, source_embeddings)
        if any(score.item() > similarity_threshold for score in cosine_scores[0]):
            return True  
    
        return False

def generate_response_for_model(query, session_id, llm_model_id, index_path):    
    
    print(f"Index file path: {index_path}")
    
    #global instance_llm
    global instance_retriever
    
    instance_llm = qna_llm(llm_model_id)
    
    print(f'using lm model {instance_llm.repo_id}')
    if not instance_retriever:
        instance_retriever = rag_retriever_hf(index_path)

    #chat_index_retriever = rag_retriever(index_path)

    print('retriever loaded')
    
    history_aware_retriever = create_history_aware_retriever(instance_llm, instance_retriever, contextualize_q_prompt)

    question_answer_chain = create_stuff_documents_chain(instance_llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    #print(type(rag_chain))
    #response = rag_chain.invoke({'input':query, "chat_history":[]})
    response = get_response(rag_chain, query, session_id)
        
    print(f'AI Answer: {response}')        
       
    #return response["answer"]
    return response

def generate_response_for_model_hf(query, session_id, llm_model_id, embedding_model, index_path):    
    
    print(f"Index file path {index_path}")
    global embedding_model_id
    
    embedding_model_id = embedding_model
    
    llm = qna_llm(llm_model_id)

    chat_index_retriever = rag_retriever_hf(index_path)

    history_aware_retriever = create_history_aware_retriever(llm, chat_index_retriever, contextualize_q_prompt)

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    #response = chatbot(user_input)[0]['generated_text']
    conversational_rag_chain = RunnableWithMessageHistory(rag_chain, get_session_history, input_messages_key="input", 
                                                          history_messages_key="chat_history", 
                                                          output_messages_key="answer",)

    try:
        
        response = conversational_rag_chain.invoke({"input": query}, 
                                        config={ "configurable": {"session_id": session_id}},  # constructs a key "abc123" in `store`.
                                        )
        
        response.keys()                                
        print(f'Original answer: {response["answer"]}')
        #is_valid_answer = validate_answer_against_sources(response["answer"], response["context"])
        
        #if not is_valid_answer:
        #    response['answer'] = "Sorry I cannot answer this question based on the knowledge base I am trained on."
        
        print(response["context"])
        print("*********************************")
        print(f'AI Answer: {response["answer"]}')        
    except Exception as e:
        print(f"Error invoking LLM {e}")
        traceback.print_exc()
        response = {"answer":"Oops, some problem handling request. Apologies for inconvenience"}
       
    return response

def generate_response(query, session_id):
    
    os.environ["OPENAI_API_KEY"] = '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'
    
    hfapi_key = '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' #getpass("Enter you HuggingFace access token:")
    os.environ["HF_TOKEN"] = hfapi_key
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hfapi_key
    
    llm_model = "HuggingFaceH4/zephyr-7b-beta"
    #embedding_model = "jinaai/jina-embeddings-v3"
    embedding_model = "flax-sentence-embeddings/all_datasets_v4_MiniLM-L6"
    #llm_model = "mistralai/Mistral-7B-Instruct-v0.2"
    #llm_model = "AdaptLLM/finance-LLM"
    #llm_model = "meta-llama/Llama-3.2-1B"
    
    config_reader.load_config()
    
    index_path = config_reader.get_property('local', 'financial_data_index')
    
    #findata = generate_response_for_model(query, session_id, llm_model, embedding_model, index_path)
    findata = generate_response_for_model(query, session_id, llm_model, index_path)
    
    #index_path = config_reader.get_property('local', 'personal_finance_index')
    #personal_findata = generate_response_for_model(query, session_id, llm_model, embedding_model, index_path)
    
    data = "Output using finance data:\n"
    data += findata
    #data += '\n****************************************************\n'
    #data += "Output using personal finance data:\n"
    #data += personal_findata
    
    return data
    
    