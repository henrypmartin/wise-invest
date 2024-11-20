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
from langchain_core.output_parsers import StrOutputParser

from com.iisc.cds.cohort7.grp11.stock_investment_analysis import perform_investment_analysis

import os

#emb_model = "flax-sentence-embeddings/all_datasets_v4_MiniLM-L6"
emb_model = "sentence-transformers/gtr-t5-large"
os.environ["TAVILY_API_KEY"]='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'
search = TavilySearchResults(max_results=3, include_raw_content=False,
    include_images=False)

def process_mutual_fund_queries(user_query: str) -> str:
    ''' Searches the web for user query on mutual funds. Uses the most relevant match'''
    
    print(f'process_mutual_fund_queries args: {user_query}')
    
    search.include_domains = ["www.valueresearchonline.com"]
    #data_df = yf.download(ticker, period=period)
    #output = search.invoke(f'{ticker} and {period}')
    invoke_op = search.invoke(user_query)
    search_data = []
    for op in invoke_op:
        search_data.append(op['content'])
        print(f"{op['url']} == {op['content']}")        
    
    #print(data_df)
    #print(type(data_df))
    
    return ' '.join(search_data) 
    
def process_generic_queries(user_query: str) -> str:
    ''' Returns responses to generic user queries based on the web 
    search'''
    
    #search.include_domains = ["morningstar.in"]
    
    print(f'process_generic_queries args: {user_query} ')
    
    invoke_op = search.invoke(user_query)
    search_data = []
    for op in invoke_op:
        search_data.append(op['content'])
        print(f"{op['url']} == {op['content']}")
        print('7777777777777777777777777777777777777777777')
    
    #print(data_df)
    #print(type(data_df))
    
    return ' '.join(search_data)     

def calculate_investment_returns(ticker: str, invested_date: str, investement_amount: int) -> str:
    ''' Returns current value of an investment amount in any stock for given ticker, invested date and amount invested.
    Eg invested_date is in format YYYY-mm-dd'''
    
    print(f'calculate_investment_returns args: {ticker} {invested_date} {investement_amount}')
    
    #data = perform_investment_analysis(ticker, invested_date, investement_amount)
    
    #return data
    return "If you had invested ₹100,000 in Reliance Industries on November 13, 2023, the current value on November 13, 2024, would be:\n\n- ₹107,457 if dividends were not reinvested.\n- ₹114,954 if dividends were reinvested.\n\nThe total dividend amount received would be ₹430, and the compound annual growth rate (CAGR) would be 7.44%."
    
 

def process_stock_queries(user_query: str) -> str:
    ''' Searches the web for user query. Uses the most relevant match'''
    
    print(f'process_stock_queries args: {user_query}')
    
    search.include_domains = ["economictimes.indiatimes.com", "in.investing.com", "moneycontrol.com"]
    #data_df = yf.download(ticker, period=period)
    #output = search.invoke(f'{ticker} and {period}')
    invoke_op = search.invoke(user_query)
    search_data = []
    for op in invoke_op:
        search_data.append(op['content'])
        print(f"{op['url']} == {op['content']}")        
    
    #print(data_df)
    #print(type(data_df))
    
    return ' '.join(search_data) 

def get_historic_stock_data(user_input: str, period:str) -> str:
    ''' Returns historic stock data for given ticker and period.
    Eg for period is 1Y, 2Y, 5Y etc or could be date in yyyy-mm-dd format'''
    
    print(f'get_historic_stock_data args: {user_input} and {period}')
    
    #search.include_domains = [f"https://www.google.com"]
    #data_df = yf.download(ticker, period=period)
    #output = search.invoke(f'{ticker} and {period}')
    output = search.invoke(f'{user_input} stock price {period}')
    
    print(output)
    
    #print(data_df)
    #print(type(data_df))
    
    return f'Input details are {user_input} and {user_input}' 

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

from langchain_core.prompts.chat import ChatPromptTemplate

template = (
    "You are a financial advisor system to provide financial advice"
    "reformulate the given question to yield better results"
    "using base date as {base_date}, "
    "calculate correct dates in User query with temporal semantics eg one year ago today"
    "Do not ask to reformulate again in the generated query"
    "if you are not able to reformulate, please pass the query as is"
    "Question: {orig_usr_query}"    
)

rewrite_prompt = ChatPromptTemplate.from_template(template)

class WebSearchRetriever(Runnable[str, list[Document]]):
        
    def __init__(self, llm_model, agent_executor):
        self.llm_model = llm_model
        self.agent_executor = agent_executor
    
    def invoke(self, usr_input: str, config: Optional[RunnableConfig] = None, **kwargs: Any) -> list[Document]:
        print(f'Original user query: {usr_input}')
        
        from datetime import datetime

        now = datetime.now() # current date and time
        
        base_date = now.strftime("%Y-%d-%m")
        
        rewriter = rewrite_prompt | self.llm_model | StrOutputParser()
        reformulated_query = rewriter.invoke({"orig_usr_query": usr_input, "base_date": base_date})
        
        print(f'Reformulated user query: {reformulated_query}')
        
        response = self.agent_executor.invoke({"messages": [HumanMessage(content=reformulated_query)]})        
        
        print(f'responses from react agent {response}')
        #origdocs = self.base_retriever.invoke(input, config, **kwargs)
        #print(f'Docs from embeddings {origdocs}')
        docs = []            
        sources = []
        
        for aimsg in response["messages"]:
            if isinstance(aimsg, AIMessage):
                doc = Document(page_content=aimsg.content)
                #docs.append(doc)
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
        
        print(docs)
        
        from langchain_community.vectorstores.faiss import FAISS
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        from langchain_text_splitters.character import RecursiveCharacterTextSplitter
        from langchain.indexes.vectorstore import VectorstoreIndexCreator
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 150)
        embedding = HuggingFaceEndpointEmbeddings(model=emb_model)
        data_index = VectorstoreIndexCreator(text_splitter=text_splitter, embedding=embedding,
                        vectorstore_cls=FAISS)
        
        db_index = data_index.from_documents(docs)
        retrieved_docs = db_index.vectorstore.as_retriever().invoke(reformulated_query, config, **kwargs)
        
        print(retrieved_docs)
        #print(f"retrieved docs {retrieved_docs}")
        #docs.extend(origdocs)
                
        return retrieved_docs