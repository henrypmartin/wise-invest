def embed_data_ollama(data_splits, data_indexer):
    
    print(f"using embedding model: {embedding_model_id}")
    
    embed_model = OllamaEmbeddings(model="mxbai-embed-large")
    #embed_model = HuggingFaceEndpointEmbeddings(repo_id=embedding_model_id)
    #embed_model = OpenAIEmbedding(model_name="FinLang/investopedia_embedding")
    #embed_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 150)
        
    #index_path = config_reader.get_property('local', 'index_dir')
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

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:    
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def validate_answer_against_sources(response_answer, source_documents):
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
