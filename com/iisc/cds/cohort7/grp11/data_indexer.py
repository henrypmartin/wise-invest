'''
Created on 19-Jul-2024

@author: Henry Martin
'''
# Get a list of all files in the directory
import os
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document
from com.iisc.cds.cohort7.grp11 import config_reader
import shutil

embedding_model_id = None

def prepare_and_embed_webscrapped_data():
    fin_articles_dir = config_reader.get_property('local', 'fin_articles_dir')
    all_dirs = os.listdir(fin_articles_dir)
    
    for dir_name in all_dirs:
        dir_name = os.path.join(fin_articles_dir, dir_name)
        prepare_and_embed_webscrapped_data_files(dir_name)
        
    
def prepare_and_embed_webscrapped_data_files(sub_dir):
    all_files = os.listdir(sub_dir)
    
    print(f'Indexing: {sub_dir}')
    
    os.makedirs(os.path.join(sub_dir, 'indexed'), exist_ok=True) 
    
    for file_name in all_files:
        print(f'file: {file_name}')
        if file_name.endswith('.txt'):
            docs = []
            
            src_file = os.path.join(sub_dir, file_name)
            
            with open(src_file, 'r', encoding='utf-8') as file:
                doc = Document(page_content=file.read(),
                               metadata={"name": file_name})                
                
            docs.append(doc)
            
            data_indexer = lambda data_index, data_splits: data_index.from_documents(data_splits)
            
            embed_data(docs, data_indexer)
            
            shutil.move(src_file, os.path.join(sub_dir, 'indexed' , file_name))
    
    
def prepare_and_embed_pdf_data():
    data_file_path = config_reader.get_property('local', 'extract_dir')
    all_files = os.listdir(data_file_path)

    print(f'Extract path: {data_file_path}')
    for file in all_files:
        print(f'file: {file}')
        if file.endswith('.pdf'):
            
            pdf_loader = PyPDFLoader(data_file_path + "/" + file)
            docs = pdf_loader.load()
            
            print(docs[0].metadata)
            
            data_indexer = lambda data_index, data_splits: data_index.from_loaders(data_splits)
            
            embed_data(docs, data_indexer)        

def embed_data(data_splits, data_indexer):
    
    print(f"using embedding model: {embedding_model_id}")
    
    embed_model = HuggingFaceEndpointEmbeddings(repo_id=embedding_model_id)
    #embed_model = OpenAIEmbedding(model_name="FinLang/investopedia_embedding")
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 150)
        
    index_path = config_reader.get_property('local', 'index_dir')
    
    if os.path.isfile(os.path.join(index_path, "index.faiss")):
        index = FAISS.load_local(index_path, embed_model, allow_dangerous_deserialization=True)
        index.add_documents(data_splits)
        index.save_local(index_path)
    else:
        print("Creating vector Db")
        data_index = VectorstoreIndexCreator(text_splitter=text_splitter, embedding=embed_model,
                            vectorstore_cls=FAISS)
    
        print("loading files")
        db_index = data_indexer(data_index, data_splits)
        db_index.vectorstore.save_local(index_path)
    
    print("complete")

def process_data(embedding_model, data_type):
    
    config_reader.load_config()
    
    global embedding_model_id
    
    embedding_model_id = embedding_model
    
    if 'pdf' == data_type:
        prepare_and_embed_pdf_data()
    elif 'webscrapped' == data_type:
        prepare_and_embed_webscrapped_data()
        
#os.environ["CONFIG_PATH"] = "C:\\Henry\\Workspace\\wise-invest\\config.properties"
#config_reader.load_config()
#prepare_and_embed_webscrapped_data()
