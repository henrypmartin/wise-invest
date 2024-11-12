'''
Created on 11-Nov-2024

@author: Henry Martin
'''
from langchain_core.tools import BaseTool
import requests
import yfinance as yf

from langchain_core.messages.tool import ToolCall
from typing import Optional, Any, Union
import json
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults

from com.iisc.cds.cohort7.grp11.stock_investment_analysis import perform_investment_analysis

import os

#emb_model = "flax-sentence-embeddings/all_datasets_v4_MiniLM-L6"
emb_model = "sentence-transformers/gtr-t5-large"

def get_mutual_fund_related_queries(user_query: str) -> str:
    ''' Returns responses to user queries on mutual funds based on the web 
    search'''
    
    os.environ["TAVILY_API_KEY"]='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'
    search = TavilySearchResults(max_results=10)
    search.include_domains = ["morningstar.in"]
    
    #output = search.run(user_query)
    output = search.invoke(user_query)
    
    print(f'get_mutual_fund_related_queries args: {user_query} and output {output}')
    
    #yf_data = yf.Ticker(ticker)
    return f'Input details are {user_query} ' 

def calculate_investment_returns(ticker: str, invested_date: str, investement_amount: int) -> str:
    ''' Returns current value of an investment amount in any stock for given ticker, invested date and amount invested.
    Eg invested_date is in format YYYY-mm-dd'''
    
    print(f'calculate_investment_returns args: {ticker} {invested_date} {investement_amount}')
    
    data = perform_investment_analysis(ticker, invested_date, investement_amount)
    
    #yf_data = yf.Ticker(ticker)
    return data 

def get_historic_stock_data(ticker: str, period: str) -> str:
    ''' Returns historic stock data for given ticker and period.
    Eg for period is 1Y, 2Y, 5Y etc or could be date in yyyy-mm-dd format'''
    
    print(f'get_historic_stock_data args: {ticker} and {period}')
    
    #yf_data = yf.Ticker(ticker)
    return f'Input details are {ticker} and {period}' 

class OriginalQueryTool(BaseTool):
    
    name: str = "custom_tool_to_convert_original_query_to_sql_query"
    description: str = (
        "Tool to convert original user input into sql query"            
        "Below are the tables and its columns: "
        "Table: DailyStockHistory, columns[Ticker, Date, Close_Price, Adjusted_Close_Price] "
    )
    
    def _run(self, query: str, **kwargs: Any) -> Any:
        print(f'Query is: {query}')
        
        output = {
            "company": f"{query}"                
            }
        
        return json.dumps(output, indent=2)
        
class YahooFinanceAPITool(BaseTool):
        
        name: str = "custom_yahoo_finance_api_tool"
        description: str = (
            "Tool to get details from yahoo finance api. "            
        )
        
        def get_ticker(self, company_name):
            url = "https://query2.finance.yahoo.com/v1/finance/search"
            headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36',
            'Referer': 'https://www.google.com',
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Cache-Control": "max-age=0",
            }
            params = {"q": company_name, "quotes_count": 1, "country": "India"}
        
            res = requests.get(url=url, params=params, headers=headers)
            data = res.json()
            company_code = data['quotes'][0]['symbol']
            return company_code
        
        def invoke(self, input: Union[str, dict, ToolCall], config: Optional[RunnableConfig] = None, **kwargs: Any,) -> Any:
            print(f'Input in invoke: {input}')
            print(f'Config in invoke: {config["configurable"]["__pregel_read"].args[1]["channel_values"]["messages"][0].content}')
            print(type(config["configurable"]["__pregel_read"]))
            print(dir(config["configurable"]["__pregel_read"]))
            
            original_user_input = config["configurable"]["__pregel_read"].args[1]["channel_values"]["messages"][0]
            tool_call_id = config["configurable"]["__pregel_read"].args[1]["channel_values"]["messages"][1].additional_kwargs['tool_calls'][0]['id']
            
            print(f'kwargs in invoke: {kwargs}')
            
            yfindata = super().invoke(input, config, **kwargs)
            
            #print(yfindata.content)
            #print(type(yfindata.content))
            
            import re
            doc_data = re.sub(r'  +|[{}"\n]', '', yfindata.content)
            #doc_data = yfindata.content.replace('\n', '').replace('\"', '').replace('      ', '')
            
            print(f"final doc data {doc_data}")
            doc = Document(page_content=doc_data, metadata = {"name": "data from yahoo finance"})
            
            from langchain_community.vectorstores.faiss import FAISS
            from langchain_huggingface import HuggingFaceEndpointEmbeddings
            from langchain_text_splitters.character import RecursiveCharacterTextSplitter
            from langchain.indexes.vectorstore import VectorstoreIndexCreator
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 150)
            embedding = HuggingFaceEndpointEmbeddings(model=emb_model)
            data_index = VectorstoreIndexCreator(text_splitter=text_splitter, embedding=embedding,
                            vectorstore_cls=FAISS)
            
            print(f'doc to indexing {doc}')
            db_index = data_index.from_documents([doc])
            print(f'indexing done {type(original_user_input.content)}')
            retrieved_docs = db_index.vectorstore.as_retriever().invoke(original_user_input.content)
            
            print(f'retrieved docs from invoke {retrieved_docs}')
            
            return ToolMessage(content=retrieved_docs, tool_call_id=tool_call_id)
            
        def _run(self, query: str, **kwargs: Any) -> Any:
            print(f'other args {kwargs}')
            print(f'Query received for custom YahooFinanceAPITool: {query} ')
            ric = self.get_ticker(query)
            
            print(f'Getting data for {ric}')
            data_df = yf.download(ric, period='5y')
            data_df.reset_index(inplace=True)
            data_df['DateYYYYMMDD'] = data_df['Date'].dt.strftime('%Y-%m-%d')
            data_df.drop(columns=['Date', 'High', 'Low', 'Open', 'Volume'], inplace=True)
            
            data = data_df.to_dict(orient='records')
            output = {
                "company": f"{query}",
                "ticker": f"{ric}",
                "historical_data": []
                }            

            # Iterate over each record in data and extract required fields
            for record in data:
                date = record[('DateYYYYMMDD', '')]
                adj_close = record[('Adj Close', ric)]
                close = record[('Close', ric)]
                
                # Append the structured daily data to the historical_data list
                output["historical_data"].append({
                    "date": date,
                    "adj_close": adj_close,
                    "close": close
                })
            
            # Convert the dictionary to a JSON string for display (optional)
            json_output = json.dumps(output, indent=2)
            #print(json_output)
            return json_output

class WebSearchRetriever(Runnable[str, list[Document]]):
        
    def __init__(self, retriever, agent_executor):
        self.base_retriever = retriever
        self.agent_executor = agent_executor
    
    def invoke(self, input: str, config: Optional[RunnableConfig] = None, **kwargs: Any) -> list[Document]:
        print(f'user query is {input}')
        response = self.agent_executor.invoke({"messages": [HumanMessage(content=input)]})
        
        print(f'responses from react agent {response}')
        #origdocs = self.base_retriever.invoke(input, config, **kwargs)
        #print(f'Docs from embeddings {origdocs}')
        docs = []            
        sources = []
        
        for aimsg in response["messages"]:
            if isinstance(aimsg, AIMessage):
                doc = Document(page_content=aimsg.content)
                docs.append(doc)
            elif isinstance(aimsg, ToolMessage):
                print(f'tool message content: {aimsg.content}')
                doc = Document(page_content=aimsg.content)
                docs.append(doc)
                #if aimsg.content:
                #    contentlst = ast.literal_eval(aimsg.content)
                #    for cntdt in contentlst:
                #        sources.append(cntdt["url"])
                #        print(cntdt["url"])
        
        #for doc in docs:
        #    doc.metadata = {"name": ",".join(sources)}
        
        from langchain_community.vectorstores.faiss import FAISS
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        from langchain_text_splitters.character import RecursiveCharacterTextSplitter
        from langchain.indexes.vectorstore import VectorstoreIndexCreator
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 150)
        embedding = HuggingFaceEndpointEmbeddings(model=emb_model)
        data_index = VectorstoreIndexCreator(text_splitter=text_splitter, embedding=embedding,
                        vectorstore_cls=FAISS)
        
        db_index = data_index.from_documents(docs)
        retrieved_docs = db_index.vectorstore.as_retriever().invoke(input, config, **kwargs)
        
        #print(f"retrieved docs {retrieved_docs}")
        #docs.extend(origdocs)
        
        return retrieved_docs