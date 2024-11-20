'''
Created on 03-Sep-2024

@author: Henry Martin
'''
import os
from typing import Sequence
from typing_extensions import Annotated, TypedDict

from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages

from com.iisc.cds.cohort7.grp11.deprecated.advisor_service_huggingface import HuggingFaceAdvisorService
from com.iisc.cds.cohort7.grp11.advisor_service_openai import OpenAIAdvisorService
from com.iisc.cds.cohort7.grp11.advisor_tools import (
WebSearchRetriever, get_historic_stock_data, calculate_stock_investment_returns,
process_company_queries, process_generic_queries, process_mutual_fund_queries)
from langgraph.checkpoint.memory import MemorySaver

from langgraph.managed.is_last_step import RemainingSteps
from langchain_core.output_parsers import StrOutputParser

agent_executor_prompt_template = (
    "You are a financial advisor system to provide financial advice"
    "reformulate the given question to yield better results"
    "using today's date as {todays_date}, calculate correct dates in User query with temporal semantics eg one year ago today etc"                
    "using today's date as {todays_date}, identify any dates specified in the user messages whether in past or in future"
    "Remove any prefix from the amount indicating currency code"
    "Prefix any amount with invested_amount:"
    "Prefix any date with invested_date:"
    "Extract the ticker symbol, company names, mutual fund names, investment amount, investment date from the user query as applicable"
    "Identify correct arguments ticker, invested_date and invested_amount from user messages to calculate_stock_investment_returns tool"
    "use the invested_date and invested_amount from query as arguments to calculate_stock_investment_returns"
    "Do not ask to reformulate again in the generated query"
    "Do not add any other prefix to the query"
    "if you are not able to reformulate, please pass the query as is"
    "Don't combine arguments for tools as single string"
    "Split arguments ticker, invested_date and invested_amount to match correct method signature for calculate_stock_investment_returns tool"
    "You have access to the following tools: {tools}"
    """calculate_stock_investment_returns : Tool to answer any queries related to current value of invested amount in stocks and investment date in the past.
                                            Args:
                                                ticker: the ticker symbol for listed Indian company
                                                invested_date: invested date in YYYY-mm-dd format 
                                                invested_amount: invested amount"""
    """process_mutual_fund_queries : Tool to answer any queries related to mutual funds
                                     Args:
                                        user_query: the query from user on mutual funds"""
    """process_company_queries : ool to answer queries related to any listed companies in India
                                     Args:
                                        user_query: the query from user on any listed companies in India"""
    "Use the following format:"
    "Question: the input question you must answer"    
    "Thought: you should always think about what to do"
    "Action: the action to take, should be one of [{tool_names}]"
    "Action Input: the input to the action"
    "Observation: the result of the action"
    "....(this Thought/Action/Action Input/Observation can repeat N times)"    
    "Thought: I now know the final answer"
    "Final Answer: the final answer to the original input question"
    "Begin!"
    "Question: {messages}"
    "Thought: {agent_scratchpad}"
)

agent_executor_prompt = ChatPromptTemplate.from_template(agent_executor_prompt_template)

user_prompt_template = (
    "You are a financial advisor system to provide financial advice"
    "You have access to the following tools:"
    """calculate_stock_investment_returns(ticker: str, invested_date: str, invested_amount: int) : Tool to answer any queries related to current value of invested amount in stocks and investment date in the past.
                                            Args:
                                                ticker: the ticker symbol for listed Indian company
                                                invested_date: invested date in YYYY-mm-dd format 
                                                invested_amount: invested amount"""
    """process_mutual_fund_queries : Tool to answer any queries related to mutual funds
                                     Args:
                                        user_query: the query from user on mutual funds"""
    """process_company_queries : ool to answer queries related to any listed companies in India
                                     Args:
                                        user_query: the query from user on any listed companies in India"""
    "Extract the ticker symbol, company names, mutual fund names, investment amount, investment date from the user query as applicable"
    "Don't combine arguments for tools as single string"
    "Split arguments ticker, invested_date and invested_amount to match correct method signature for calculate_stock_investment_returns tool"
    "using today's date as {todays_date}, calculate correct dates in User query with temporal semantics eg one year ago today etc"                
    "using today's date as {todays_date}, identify any dates specified in the user messages whether in past or in future"
    "Question: {messages}"
)

user_query_prompt = ChatPromptTemplate.from_template(user_prompt_template)

user_query_reformatter_template = (
    "You are a financial advisor system to provide financial advice"
    "reformulate the given question to yield better results"
    "using today's date as {todays_date}, calculate correct dates in User query with temporal semantics eg one year ago today etc"                
    "using today's date as {todays_date}, identify any dates specified in the user messages whether in past or in future"
    "The ticker symbol for listed Indian company in the form as expected by Yahoo finance API eg INFY.NS for Infosys limited"
    "Remove any prefix from the amount indicating currency code"
    "Prefix any amount with invested_amount:"
    "Prefix any date with invested_date:"
    "use the data from reformatted query as arguments to appropriate tools"
    "Do not ask to reformulate again in the generated query"
    "Do not add any other prefix to the query"
    "if you are not able to reformulate, please pass the query as is"
    "The queries should be limited to Indian context"
    "Add prefix 'Todays date is {todays_date}.' to the reformatted query"
    "Use below chat history for historical chat context:"
    "Chat History: {chat_hist}"
    "Question: {messages}"    
)

rewrite_prompt = ChatPromptTemplate.from_template(user_query_reformatter_template)

stock_fundatentals_template = ''' 
1. Introduction
    Briefly describe the company, its business model, and industry. Include details such as market position, main products or services, and geographical presence.
2. Financial Metrics
    Revenue and Growth Trends: provide numbers for annual revenue. Figures are in crores. Comment on the growth or decline trends.
    Profitability:
        Net Profit Margin: provide numbers for Ratio of net income to revenue.
        Operating Margin: provide numbers for Operating profit as a percentage of revenue.
        Return Metrics:
            ROE (Return on Equity): <Profitability relative to shareholders’ equity> provide numbers for return on equity.
            ROA (Return on Assets): <Efficiency in using assets to generate profits> provide numbers for return on assets.
            Earnings Per Share (EPS): provide numbers for earnings per share.
            Dividends: provide numbers for dividend payout history, yield, and growth.
3. Valuation Metrics
    Price-to-Earnings (P/E) Ratio: <Assess if the stock is overvalued or undervalued> provide numbers for price to earnings ratio.
    Price-to-Book (P/B) Ratio: <Compare market value with book value> provide numbers for price to book ratio.
    PEG Ratio: <Evaluate P/E relative to earnings growth> provide numbers for PEG ratio.
4. Liquidity and Solvency
    Current Ratio: <Current assets divided by current liabilities> provide numbers for current ratio. Comment on the current ratio numbers 
    Debt-to-Equity Ratio: <Indication of financial leverage> provide numbers for debt to equity ratio. Comment on the Debt-to-equity ratio numbers
    Interest Coverage Ratio: <Ability to cover interest expenses> provide numbers for interest coverage ratio.
5. Operating Performance
    Highlight inventory turnover, asset turnover, and other efficiency metrics.
    Compare against industry averages or competitors.
6. Sector and Industry Analysis
    Contextualize the company's performance with broader industry trends.
    Include growth drivers, potential challenges, and regulatory impacts.
7. Management and Governance
    Evaluate the management team, strategy, and corporate governance practices.
8. SWOT Analysis
    Summarize strengths, weaknesses, opportunities, and threats.
9. Investment Perspective
    Conclude with key takeaways:
    Is the company undervalued or overvalued?
    Long-term growth potential or risks.
'''

output_formatter_prompt_template = (
    "You are a financial advisor system to provide financial advice"
    "Present and structure the answer in more readable and easy to follow format"
    "When presenting the data, please use correct and factual dates in the output "
    "for eg if mentioning a share price, provide a date of share price or if mentioning financial results, provide a date of results"
    "Do not include unnecessary information like Vigilance Awareness or IPO Details etc"
    f"If the question is on fundamentals of any company, please use format {stock_fundatentals_template}"
    "Include the source of data in the response"
    "Suggest related follow-up questions"    
    )

from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.errors import GraphRecursionError

from langgraph.prebuilt import create_react_agent

memory = MemorySaver()

def get_agent_executor():
    advisor_service = OpenAIAdvisorService()
    
    instance_llm = advisor_service.qna_llm()
    tools = [process_company_queries, calculate_stock_investment_returns, process_mutual_fund_queries]
    
    agent_executor = create_react_agent(instance_llm, tools, state_modifier=output_formatter_prompt_template, checkpointer=memory, debug=True)
    
    return agent_executor, instance_llm

agent_executor, instance_llm = get_agent_executor()

def generate_response(query, session_id):
    
    #hf_srvc = HuggingFaceAdvisorService()
    
    config = {"configurable": {"thread_id": session_id}}
    from datetime import datetime

    now = datetime.now() # current date and time
    
    base_date = now.strftime("%Y-%m-%d")
    
    mem_state = memory.get(config=config)
    
    chat_hist = []
    if mem_state:
        mem_messages = mem_state['channel_values']['messages']
        for msg in mem_messages:
            if not isinstance(msg, ToolMessage) and msg.content:
                chat_hist.append({msg.__class__.__name__:msg.content})
                
    
    print(f'Chat history: {chat_hist}')
    rewriter = rewrite_prompt | instance_llm | StrOutputParser()
    reformulated_query = rewriter.invoke({"messages": query, "todays_date": base_date, "chat_hist":chat_hist[-4:]})
    
    print(f"Reformatted query: {reformulated_query}, userid: {session_id}")
    #custom_retriever = get_web_results(instance_llm, None)
    
    #response = custom_retriever.invoke(query, None)
    
    class CustomState(AgentState):
        todays_date: str
        #remaining_steps: RemainingSteps
        
    #instance_llm = advisor_service.qna_llm()
    from langgraph.prebuilt import create_react_agent    
    #from langchain.agents import create_react_agent
    
    def _modify_state_messages(state: CustomState):
        print(f"Current state: {state}")
        # Give the agent amnesia, only keeping the original user query
        if state["remaining_steps"] <= 23:
            state["is_last_step"] = True
            return "__end__"
        
        return user_query_prompt.invoke({"messages": state["messages"], "todays_date": state["todays_date"]})

        
    #from langchain.agents import AgentExecutor
    
    #agent_executor = create_react_agent(instance_llm, tools=tools, prompt=agent_executor_prompt)
    #agent_executor = AgentExecutor(agent=agent_executor, tools=tools, handle_parsing_errors=True)
    #agent_executor.set_verbose(True)
    
    intermediate_steps = []
    
    #response = agent_executor.invoke({"messages": query, "todays_date": base_date}, None)
    #response = agent_executor.invoke({"messages": query, "todays_date": base_date, "intermediate_steps": intermediate_steps}, None)
    response = agent_executor.invoke({"messages": reformulated_query, "todays_date": base_date}, config=config)
    
    print(f'Response: {response}')
    #print(dir(response))
    #response.       
        
    ai_response = []
    
    for aimsg in response["messages"]:
        if isinstance(aimsg, HumanMessage):
            ai_response.clear()
            
        if isinstance(aimsg, AIMessage) and aimsg.content:
            ai_response.append(aimsg.content)            
    
    final_output = '\n'.join(ai_response)
    
    print(f"Financial Advisor:{final_output}")
       
    return f"Financial Advisor:{final_output}"

#generate_response('what are fundamentals of balaji amines?', 1)
#generate_response('Suggest best large cap mutual funds in India', 1)
#generate_response('what would have been the value of 100000 invested in reliance industries and infosys limited on 2023-11-16 be today', 1)
#generate_response('what would have been the value of 100000 invested in reliance industries on 2023-11-16 be today', 1)
#generate_response('what would have been the value of Rs 100000 invested in reliance industries one year ago be today', 1)
#generate_response('what is the latest share price of reliance industries and as of what date?', 1)
#generate_response('what are the best growth stocks in India?', 1)
#generate_response('i am 40 years old, how much do i have to invest to get pension of Rs 100000 per month after retirement', 1)