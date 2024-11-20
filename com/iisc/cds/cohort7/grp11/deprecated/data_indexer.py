'''
Created on 19-Jul-2024

@author: Henry Martin
'''
# Get a list of all files in the directory
import os
#from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
#from langchain.indexes.vectorstore import VectorstoreIndexCreator
#from langchain_community.vectorstores.faiss import FAISS
#from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain_community.vectorstores.faiss import FAISS

from com.iisc.cds.cohort7.grp11 import config_reader
import shutil
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.simple import SpaceTokenizer
import html

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
tknzr = SpaceTokenizer()

embedding_model_id = None

index_dir = None

def data_cleaning_and_preprocessing(text):
    text = html.unescape(text)
    
    txt = tknzr.tokenize(text)
    
    #remove words made up of only special chars 
    txt = [wrd.lstrip().rstrip() for wrd in txt if (not re.fullmatch('[^A-Za-z0-1]+', wrd.lstrip().rstrip())) and len(wrd.lstrip().rstrip()) > 0]
    
    #remove stopwords
    txt = [word for word in txt if word not in stop_words]
    
    #lemmetize
    txt = [lemmatizer.lemmatize(word) for word in txt]
    
    return " ".join(txt)

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
                file_text = file.read()
                len_before = len(file_text)
                file_text = data_cleaning_and_preprocessing(file_text)
                len_after = len(file_text)
                print(f"Length before {len_before} after: {len_after}")
                doc = Document(page_content=file_text,
                               metadata={"name": file_name})
                
            docs.append(doc)
            
            data_indexer = lambda data_index, data_splits: data_index.from_documents(data_splits)
            
            embed_data(docs, data_indexer)
            
            shutil.move(src_file, os.path.join(sub_dir, 'indexed' , file_name))
   

def embed_data(docs, data_indexer):
    
    os.environ["OPENAI_API_KEY"] = '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'
    #print(f"using embedding model: {embedding_model_id}")
    
    from langchain_openai import OpenAIEmbeddings
    
    embedding = OpenAIEmbeddings()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 150)
        
    index_path = os.path.join(index_dir, "index.faiss")
    
    print("Creating vector Db")
    data_index = VectorstoreIndexCreator(text_splitter=text_splitter, embedding=embedding,
                            vectorstore_cls=FAISS)
    
    print("loading files")        
    db_index = data_index.from_documents(docs)
    
    #if os.path.isfile(index_path):
    #    print("Loading existing vector Db")
    #    index = FAISS.load_local(index_dir, embedding, allow_dangerous_deserialization=True)
        
    #    index.merge_from(db_index.vectorstore)
    #    print("Saving files")
    #    index.save_local(index_dir)
    #else:        
    print("Saving files")        
    db_index.vectorstore.save_local(index_dir, index_name=docs[0].metadata.get("name"))
    
    print("complete")

def embed_data_inmemory(docs, data_indexer):
    
    os.environ["OPENAI_API_KEY"] = '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'
    #print(f"using embedding model: {embedding_model_id}")
    
    from langchain_openai import OpenAIEmbeddings
    
    embedding = OpenAIEmbeddings()
    vectorstore = InMemoryVectorStore(embedding)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 150)
    splits = text_splitter.split_documents(docs)
        
    #index_path = config_reader.get_property('local', 'index_dir')
    index_path = os.path.join(index_dir, "embeddings.index")
    
    if os.path.isfile(index_path):
        print("Loading existing vector Db")
        vectorstore = vectorstore.load(index_path, embedding)
        vectorstore.add_documents(documents=splits)
        print("Saving index")
        vectorstore.dump(index_path)
    else:
        vectorstore.add_documents(documents=splits)
        print("Saving index")
        vectorstore.dump(index_path)        

    print("complete")
    
def process_data(embedding_model, data_type):
    
    config_reader.load_config()
    
    global embedding_model_id
    global index_dir
    
    embedding_model_id = embedding_model
    
    if 'pdf' == data_type:
        index_dir = config_reader.get_property('local', 'pdf_data_index')
        #prepare_and_embed_pdf_data()
    elif 'webscrapped' == data_type:
        index_dir = config_reader.get_property('local', 'personal_finance_index')
        prepare_and_embed_webscrapped_data()
    elif 'financedata' == data_type:
        index_dir = config_reader.get_property('local', 'financial_data_index')
        fin_data_dir = config_reader.get_property('local', 'fin_data_dir')
        prepare_and_embed_webscrapped_data_files(fin_data_dir)
        
        
#os.environ["CONFIG_PATH"] = "C:\\Henry\\Workspace\\wise-invest\\config.properties"
#config_reader.load_config()
#prepare_and_embed_webscrapped_data()
