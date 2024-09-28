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

index_path=None
embedding_model_id = None

def qna_llm(model_id):
    llm = HuggingFaceEndpoint(        
        repo_id=model_id,
        #task="text-generation",
        task="question-answering",
        max_new_tokens = 512,
        top_k = 30,
        temperature = 0.1,
        repetition_penalty = 1.03,
        )
    return llm

def rag_retriever():
    embeddings = HuggingFaceEndpointEmbeddings(repo_id=embedding_model_id)
    
    index = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    return index.as_retriever()

### Contextualize question ###
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "formulate a question which can be understood without the chat history."
    "Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
    
system_prompt = (
    "You are a financial advisor system for financial advice related tasks. "
    "Use the retrieved context to answer the original question." 
    "Do not fabricate information."
    "If you don't know the answer, say that you don't know." 
    "Maintain an ethical and unbiased tone, avoiding harmful or offensive content."
    "No creativity in responses."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

print("chatbot model load complete.")

store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:    
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def validate_answer_against_sources(response_answer, source_documents):
    model = SentenceTransformer('all-mpnet-base-v2')
        
    similarity_threshold = 0.9  
    source_texts = [doc.page_content for doc in source_documents]
    answer_embedding = model.encode(response_answer, convert_to_tensor=True)
    source_embeddings = model.encode(source_texts, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(answer_embedding, source_embeddings)
    if any(score.item() > similarity_threshold for score in cosine_scores[0]):
        return True  

    return False

def generate_response(query, session_id, llm_model_id, embedding_model):
    
    config_reader.load_config()
    
    global index_path
    
    index_path = config_reader.get_property('local', 'index_dir')
    
    global embedding_model_id
    
    embedding_model_id = embedding_model
    
    llm = qna_llm(llm_model_id)

    chat_index_retriever = rag_retriever()

    history_aware_retriever = create_history_aware_retriever(llm, chat_index_retriever, contextualize_q_prompt)

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    #response = chatbot(user_input)[0]['generated_text']
    conversational_rag_chain = RunnableWithMessageHistory(rag_chain, get_session_history, input_messages_key="input", 
                                                          history_messages_key="chat_history", 
                                                          output_messages_key="answer",)

    response = conversational_rag_chain.invoke({"input": query}, 
                                    config={ "configurable": {"session_id": session_id}},  # constructs a key "abc123" in `store`.
                                    )
                                    
    is_valid_answer = validate_answer_against_sources(response["answer"], response["context"])
    
    if not is_valid_answer:
        response['answer'] = "No similarity. Sorry I can not answer the question based on the given documents"
    
    retrieved_doc = response["context"]
    print(response["context"])
    print("*********************************")
    print(response["answer"])
    #print(f"[Source: {retrieved_doc.metadata['source']}#{retrieved_doc.metadata['page']}]")
    
    response = conversational_rag_chain.invoke({"input": "Suggest follow-up question to previous answer, do not answer the question"}, 
                                    config={ "configurable": {"session_id": session_id}},  # constructs a key "abc123" in `store`.
                                    )
    
    print(f"Follow-up question: {response['answer']}")
    
    return response
