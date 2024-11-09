'''
Created on 09-Nov-2024

@author: Henry Martin
'''

from langchain_community.embeddings import GPT4AllEmbeddings


model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
gpt4all_kwargs = {"allow_download": "True"}
embeddings = GPT4AllEmbeddings(model_name=model_name, gpt4all_kwargs=gpt4all_kwargs)


