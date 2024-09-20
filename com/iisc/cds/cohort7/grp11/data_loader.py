'''
Created on 19-Jul-2024

@author: Nomura
'''
# Get a list of all files in the directory
import os
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from com.iisc.cds.cohort7.grp11 import config_reader

local_extract_path = None
index_path = None
embedding_model_id = None

def prepare_and_embed_data():
    data_file_path = local_extract_path
    all_files = os.listdir(data_file_path)

    print(f'Extract path: {data_file_path}')
    for file in all_files:
        print(f'file: {file}')
        if file.endswith('.pdf'):
            
            docs = []
            docs.append(PyPDFLoader(data_file_path + "/" + file))
            
            embed_data(docs)        

def embed_data(data_splits):
    
    print(f"using embedding model: {embedding_model_id}")
    # Load FinBERT tokenizer and model
    #tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    #model = BertEmbeddings('yiyanghkust/finbert-tone')
    embed_model = HuggingFaceEndpointEmbeddings(repo_id=embedding_model_id)
    #embed_model = OpenAIEmbedding(model_name="FinLang/investopedia_embedding")
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 150)
    
    print("Creating vector Db")
    data_index = VectorstoreIndexCreator(text_splitter=text_splitter, embedding=embed_model,
                            vectorstore_cls=FAISS)
    
    print("loading files")
    db_index = data_index.from_loaders(data_splits)
    
    #faiss.write_index(db_index, file_path)
    db_index.vectorstore.save_local(index_path)
    
    print("complete")
    return db_index

def process_data(embedding_model):
    
    config_reader.load_config()
    
    global embedding_model_id
    
    embedding_model_id = embedding_model
    
    global local_extract_path
    
    local_extract_path = config_reader.get_property('local', 'extract_dir')
    
    global index_path
    
    index_path = config_reader.get_property('local', 'index_dir')
    
    prepare_and_embed_data()
    
    
#embed_data()