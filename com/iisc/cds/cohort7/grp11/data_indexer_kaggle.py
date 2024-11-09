'''
Created on 19-Jul-2024

@author: Henry Martin
'''
# Get a list of all files in the directory
import os
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document
import shutil
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

embedding_model_id = None

index_dir = None

financial_data_index='/kaggle/working/index/financial_data_index'
personal_finance_index='/kaggle/working/index/personal_finance_index'
fin_articles_dir='/kaggle/input/financial-dataset/financial_articles/financial_articles'
fin_data_dir='/kaggle/input/financial-dataset/financial_data/financial_data'

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def data_cleaning_and_preprocessing(text):
    #remove words made up of only special chars 
    txt = [wrd.lstrip().rstrip() for wrd in text if (not re.fullmatch('[^A-Za-z0-1]+', wrd.lstrip().rstrip())) and len(wrd.lstrip().rstrip()) > 0]
    
    #remove stopwords
    txt = [word for word in txt if word not in stop_words]
    
    #lemmetize
    txt = [lemmatizer.lemmatize(word) for word in text]
    
    
    

def prepare_and_embed_webscrapped_data():
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
    
    
def embed_data(data_splits, data_indexer):
    
    print(f"using embedding model: {embedding_model_id}")
    
    embed_model = HuggingFaceEndpointEmbeddings(repo_id=embedding_model_id)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 150)
        
    index_path = index_dir
    
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
    
    global embedding_model_id
    global index_dir
    
    embedding_model_id = embedding_model
    
    if 'webscrapped' == data_type:
        index_dir = personal_finance_index
        prepare_and_embed_webscrapped_data()
    elif 'financedata' == data_type:
        index_dir = financial_data_index
        prepare_and_embed_webscrapped_data_files(fin_data_dir)
