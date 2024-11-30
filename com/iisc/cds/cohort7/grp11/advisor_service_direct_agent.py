'''
Created on 03-Sep-2024

@author: Henry Martin
'''
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import RemoveMessage

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from com.iisc.cds.cohort7.grp11.advisor_prompts import rewrite_prompt, output_formatter_prompt_template
from com.iisc.cds.cohort7.grp11.advisor_tools import (calculate_stock_investment_returns, 
                                                      process_company_queries, 
                                                      process_generic_mutual_fund_queries,
                                                      process_specific_mutual_fund_queries,
                                                      run_portfolio_allocator,
                                                      process_generic_queries)

from datetime import datetime
import traceback
import logging
from logging.handlers import RotatingFileHandler
import os

#file_path = os.path.join(os.path.dirname(__file__), "../../../../", "wise_invest.log")

#handler = RotatingFileHandler(file_path, maxBytes=5000000, backupCount=5)

#logging.basicConfig(handlers=[handler],
#                    level=logging.INFO,
#                    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

memory = MemorySaver()
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

def get_agent_executor():
    
    tools = [process_company_queries, calculate_stock_investment_returns, 
             process_specific_mutual_fund_queries, process_generic_mutual_fund_queries,
             run_portfolio_allocator, process_generic_queries]
    
    agent_executor = create_react_agent(llm, tools, state_modifier=output_formatter_prompt_template, checkpointer=memory, debug=True)
        
    return agent_executor

agent_executor = get_agent_executor()

def generate_response(query, session_id):
    
    config = {"configurable": {"thread_id": session_id}}
    
    if "messages" in agent_executor.get_state(config).values:
        hist_messages = agent_executor.get_state(config).values["messages"]
        logger.info(f'messages in memory {len(hist_messages)}')
        
        #if len(hist_messages) > 4:
        msgs_to_delete = []
        for msg in hist_messages:
            if not isinstance(msg, ToolMessage):
                msgs_to_delete.append(RemoveMessage(id=msg.id))
            
        #agent_executor.update_state(config, {"messages": [RemoveMessage(id=m.id) for m in hist_messages[:-3]]})
        #agent_executor.update_state(config, {"messages": msgs_to_delete})
        
    now = datetime.now() # current date and time
    
    base_date = now.strftime("%Y-%m-%d")
    
    mem_state = memory.get(config=config)
    
    chat_hist = []
    if mem_state:
        mem_messages = mem_state['channel_values']['messages']
        for msg in mem_messages:
            if not isinstance(msg, ToolMessage) and msg.content:
                chat_hist.append({msg.__class__.__name__:msg.content})
                
    
    logger.info(f'Chat history: {chat_hist}')
    rewriter = rewrite_prompt | llm | StrOutputParser()
    reformulated_query = rewriter.invoke({"messages": query, "todays_date": base_date, "chat_hist":chat_hist[-4:]})
    
    logger.info(f"Reformatted query: {reformulated_query}, userid: {session_id}")
    
    final_output = 'Unable to retrieve data at this moment, please try again later.'
    try:
        response = agent_executor.invoke({"messages": reformulated_query, "todays_date": base_date}, config=config)
    
        ai_response = []
        
        for aimsg in response["messages"]:
            if isinstance(aimsg, HumanMessage):
                ai_response.clear()
                
            if isinstance(aimsg, AIMessage) and aimsg.content:
                ai_response.append(aimsg.content)            
        
        final_output = '\n'.join(ai_response)
    except:
        logger.error(f"Error occurred while processing request: {traceback.format_exc()}")

    logger.info(f"Financial Advisor:{final_output}")
       
    return f"Financial Advisor:{final_output}"

#generate_response('what are fundamentals of balaji amines?', 1)
#generate_response('Suggest best large cap mutual funds in India', 1)
#generate_response('what would have been the value of 100000 invested in reliance industries and infosys limited on 2023-11-16 be today', 1)
#generate_response('what would have been the value of 100000 invested in reliance industries on 2023-11-16 be today', 1)
#generate_response('what would have been the value of Rs 100000 invested in reliance industries one year ago be today', 1)
#generate_response('what is the latest share price of reliance industries and as of what date?', 1)
#generate_response('what are the best growth stocks in India?', 1)
#generate_response('i am 40 years old, how much do i have to invest to get pension of Rs 100000 per month after retirement', 1)