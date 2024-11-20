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

from langchain.agents import Tool
from langchain.tools import StructuredTool

from langchain_core.tools import tool

import os

#emb_model = "flax-sentence-embeddings/all_datasets_v4_MiniLM-L6"
emb_model = "sentence-transformers/gtr-t5-large"
os.environ["TAVILY_API_KEY"]='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'
search = TavilySearchResults(max_results=3, include_raw_content=False,
    include_images=False)


def process_mutual_fund_queries_1(user_query: str) -> str:
    '''  Tool to answer any queries related to mutual funds '''
    
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

process_mutual_fund_queries = StructuredTool.from_function(
    name='Mutual_funds_queries_answering_tool',
    func= process_mutual_fund_queries_1,
    description="Tool to answer any queries related to mutual funds"
)
    
@tool
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


def calculate_stock_investment_returns_1(ticker: str, invested_date: str, invested_amount: int) -> str:
    ''' Tool to answer any queries related to current value of invested amount in stocks and investment date in the past.
        Args:
            ticker: the ticker symbol for listed Indian company
            invested_date: invested date in YYYY-mm-dd format 
            invested_amount: invested amount'''
    
    print(f'calculate_investment_returns args: {ticker} {invested_date} {invested_amount}')
    
    #data = perform_investment_analysis(ticker, invested_date, investement_amount)
    
    #return data
    return f"If you had invested ₹100,000 in {ticker} on November 13, 2023, the current value on November 13, 2024, would be:\n\n- ₹107,457 if dividends were not reinvested.\n- ₹114,954 if dividends were reinvested.\n\nThe total dividend amount received would be ₹430, and the compound annual growth rate (CAGR) would be 7.44%."
    
 
calculate_stock_investment_returns = StructuredTool.from_function(
    name='stock_investment_current_value_queries_answering_tool',
    func= calculate_stock_investment_returns_1,
    description="""Tool to answer any queries related to current value of invested amount in stocks and investment date in the past.
                Args:
                    ticker: the ticker symbol for listed Indian company
                    invested_date: invested date in YYYY-mm-dd format 
                    invested_amount: invested amount"""
)


def process_company_queries_1(user_query: str) -> str:
    ''' Tool to answer queries related to any listed companies in India '''
    
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

process_company_queries = StructuredTool.from_function(
    name='company_related_queries_answering_tool',
    func= process_company_queries_1,
    description="Tool to answer queries related to any listed companies in India"
)

@tool
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

from langchain_core.prompts.chat import ChatPromptTemplate

template = (
    "You are a financial advisor system to provide financial advice"
    "reformulate the given question to yield better results"
    "using base date as {base_date}, "
    "calculate correct dates in User query with temporal semantics eg one year ago today"
    "Remove any prefix from the amount indicating currency code"
    "Prefix any amount with invested_amount:"
    "Prefix any date with invested_date:"
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
        
        #response = self.agent_executor.invoke({"input": reformulated_query})
        #response = self.agent_executor.invoke({"input": [HumanMessage(content=reformulated_query)]})        
        #response = self.agent_executor.invoke({"messages": [HumanMessage(content=reformulated_query)]})
        response = self.agent_executor.invoke({"messages": reformulated_query})
        
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
        
        #print(f"retrieved docs {retrieved_docs}")
        #docs.extend(origdocs)
                
        return docs