'''
Created on 03-Sep-2024

@author: Henry Martin
'''
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
#from langgraph.prebuilt import create_react_agent

from typing import Sequence
from langchain_community.tools.tavily_search import TavilySearchResults


from typing_extensions import Annotated, TypedDict
import os

from com.iisc.cds.cohort7.grp11.deprecated.advisor_service_huggingface import HuggingFaceAdvisorService
from com.iisc.cds.cohort7.grp11.deprecated.advisor_service_openai import OpenAIAdvisorService

rag_chain = None

### Contextualize question ###
contextualize_q_system_prompt = (
    "You are a financial advisor system to provide financial advice"
    "You have access to tools to fetch real-time data of stocks price, equities, company information, dividends, annual reports, etc"
    "Given a chat history and the latest user question"
    "reformulate a question which can be understood without the chat history."
    "calculate the dates correctly based based on semantic meaning of query"
    "Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
    
system_prompt = (
    "You are a financial advisor system to provide financial advice on equities and stocks. "
    "You have access to tools to fetch real-time data of stocks price, equities, company information, dividends, annual reports, corporate actions etc"
    "Use the retrieved context to answer the original question." 
    "While answering the question, take into account any corporate actions like bonuses, stock split etc that might have happened on the stock"
    "Adjust the data accordingly to these corporate actions especially stock splits and bonuses"
    "calculate the dates correctly based based on semantic meaning of query"
    "Do not fabricate information."
    "If you don't know the answer, say that you don't know." 
    "Maintain an ethical and unbiased tone, avoiding harmful or offensive content."
    "No creativity in responses."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

user_prompt_1 = (
    "You are a financial advisor system to provide financial advice"
    "calculate_stock_investment_returns : Tool to answer any queries related to current value of invested amount in stocks and investment date in the past"
    "process_mutual_fund_queries : Tool to answer any queries related to mutual funds"
    "process_company_queries : Tool to answer queries related to any listed companies in India"
    "Extract the ticker symbol, company names, mutual fund names, investment amount, investment date from the user query as applicable"
    "use the invested_date and invested_amount from query as arguments to calculate_stock_investment_returns"
    "Tools: {tools}"
    "Tool Names: {tool_names}"
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
    "Question: {input}"
    "Thought: {agent_scratchpad}"    
)

user_prompt = (
    "You are a financial advisor system to provide financial advice"
    "You have access to the following tools:"
    "calculate_stock_investment_returns : Tool to answer any queries related to current value of invested amount in stocks and investment date in the past"
    "process_mutual_fund_queries : Tool to answer any queries related to mutual funds"
    "process_company_queries : Tool to answer queries related to any listed companies in India"
    "Extract the ticker symbol, company names, mutual fund names, investment amount, investment date from the user query as applicable"
    "use the invested_date and invested_amount from query as arguments to calculate_stock_investment_returns"            
)
user_query_prompt = ChatPromptTemplate.from_template(user_prompt)

# We define a dict representing the state of the application.
# This state has the same input and output keys as `rag_chain`.
class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages] #The `add_messages` function in the annotation defines how this state key should be updated
    context: str
    answer: str

def get_response(rag_chain, query, session_id):
    
    config = {"configurable": {"thread_id": session_id}}

    result = rag_chain.invoke(
        {"input": query, "chat_history":[]},#chat_history will be updated automatically by the call_model node
        config=config,
    )
    
    print('*************************************************')
    print(result)
    print('*************************************************')
    
    return result["answer"]

from com.iisc.cds.cohort7.grp11.advisor_tools import ( 
WebSearchRetriever, get_historic_stock_data, calculate_stock_investment_returns,
process_company_queries, process_generic_queries, process_mutual_fund_queries)

def generate_response(query, session_id):
    
    advisor_service = OpenAIAdvisorService()
    
    instance_llm = advisor_service.qna_llm()
    tools = [process_company_queries, calculate_stock_investment_returns, process_mutual_fund_queries]
    
    #custom_retriever = get_web_results(instance_llm, None)
    
    #response = custom_retriever.invoke(query, None)
    
    from langgraph.prebuilt import create_react_agent    
    agent_executor = create_react_agent(instance_llm, tools)
    
    #from langchain.agents import AgentExecutor
    #from langchain.agents import create_react_agent
    
    #agent = create_react_agent(llm_model, tools=tools, prompt=user_query_prompt)
    #agent_executor = AgentExecutor(agent=agent, tools=tools)
    
    
    response = agent_executor.invoke({"messages": query}, None)
    
    print(f'Response: {response}')        
    
    ai_response = []
    
    for aimsg in response["messages"]:
        if isinstance(aimsg, AIMessage) and aimsg.content:
            ai_response.append(aimsg.content)            
    
    final_output = '\n'.join(ai_response)
    
    print(f"Financial Advisor:{final_output}")
       
    return f"Financial Advisor:{final_output}"



def generate_response_new(query, session_id):
    
    #os.environ["OPENAI_API_KEY"] = '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'
    #global rag_chain
        
    #user_query_prompt.
    response = get_response(rag_chain, query, session_id)
        
    print(f'AI Answer: {response}')        
       
    #return response["answer"]
    return response    

def get_web_results(llm_model, retriever):
    
    tools = [process_company_queries, calculate_stock_investment_returns, process_mutual_fund_queries]
    
    from langgraph.prebuilt import create_react_agent    
    agent_executor = create_react_agent(llm_model, tools, state_modifier=user_query_prompt)
    
    #from langchain.agents import AgentExecutor
    #from langchain.agents import create_react_agent
    
    #agent = create_react_agent(llm_model, tools=tools, prompt=user_query_prompt)
    #agent_executor = AgentExecutor(agent=agent, tools=tools)
    
    return WebSearchRetriever(llm_model, agent_executor)

def get_rag_chain(): 
    
    print("Initializing RAG chain")
    
    advisor_service = OpenAIAdvisorService()
    #advisor_service = HuggingFaceAdvisorService()
    
    instance_llm = advisor_service.qna_llm()
    
    print("Loaded LLM model")
    #instance_retriever = advisor_service.rag_retriever()
    instance_retriever=None
    custom_retriever = get_web_results(instance_llm, instance_retriever)
    print("Loaded RAG retriever")
    
    history_aware_retriever = create_history_aware_retriever(instance_llm, custom_retriever, contextualize_q_prompt)

    question_answer_chain = create_stuff_documents_chain(instance_llm, qa_prompt)

    ragchain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    # Define a simple node 'call_model' that runs the `rag_chain`.
    # The `return` values of the node update the graph state, so here we just
    # update the chat history with the input message and response.
    # Nodes represent units of work. They are typically regular python functions.
    def call_model(state: State):        
        
        print(f"State input {state}")
           
        response = ragchain.invoke(state, config={'arbitrary_types_allowed':True})
        return {
            "chat_history": [
                HumanMessage(state["input"]),
                AIMessage(response["answer"]),
            ],
            "context": response["context"],
            "answer": response["answer"],
        }
    
    # Our graph consists only of one node:
    workflow = StateGraph(state_schema=State)
    workflow.add_edge(START, "model") #Entry point. This tells our graph where to start its work each time we run it.
    workflow.add_node("model", call_model)# 1st arg is the unique node name and 2nd arg is the function or object that will be called whenever the node is used
    
    # Finally, we compile the graph with a checkpointer object.
    # This persists the state, in this case in memory.
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    print("RAG chain initialized")
    
    return app

rag_chain = get_rag_chain()
    
#generate_response('Suggest best large cap mutual funds in India', 1)
generate_response('what would have been the value of Rs 100000 invested in reliance industries one year ago be today', 1)
#generate_response('what is the latest share price of reliance industries and as of what date?', 1)
#generate_response('what are the best growth stocks in India?', 1)
#generate_response('i am 40 years old, how much do i have to invest to get pension of Rs 100000 per month after retirement', 1)