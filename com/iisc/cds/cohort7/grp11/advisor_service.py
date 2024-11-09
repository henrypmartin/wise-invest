'''
Created on 03-Sep-2024

@author: Henry Martin
'''
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

import pydantic
from typing_extensions import Annotated, TypedDict
from typing import Sequence

embedding_model_id = None

### Contextualize question ###
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "formulate a question which can be understood without the chat history."
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
    "You are a financial advisor system for financial advice including stocks. "
    "You have stocks price data from April 2016 to July 2019"
    "Use the retrieved context to answer the original question." 
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

# We define a dict representing the state of the application.
# This state has the same input and output keys as `rag_chain`.
class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str

def get_response(rag_chain, query, session_id):
    
    # We then define a simple node that runs the `rag_chain`.
    # The `return` values of the node update the graph state, so here we just
    # update the chat history with the input message and response.
    def call_model(state: State):        
           
        response = rag_chain.invoke(state, config={'arbitrary_types_allowed':True})
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
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)
    
    # Finally, we compile the graph with a checkpointer object.
    # This persists the state, in this case in memory.
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    config = {"configurable": {"thread_id": session_id}}

    result = app.invoke(
        {"input": query},
        config=config,
    )
    
    print('*************************************************')
    print(result)
    print('*************************************************')
    
    return result["answer"]