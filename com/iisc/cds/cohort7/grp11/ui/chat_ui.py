'''
Created on 05-Oct-2024

@author: Henry
'''
import streamlit as st

from com.iisc.cds.cohort7.grp11.advisor_service_direct_agent import generate_response

st.set_page_config(page_title="Wise-Invest")

display_title = "Wise-Invest your personal financial advisor"

st.markdown(display_title)

if 'chathistory' not in st.session_state:
    st.session_state.chathistory = []

for message in st.session_state.chathistory:
    with st.chat_message(message['role']):
        st.markdown(message['text'])

input_text = st.chat_input("I am your personal financial advisor. How can I help you today?")
               
if 'fileids' not in st.session_state:
    st.session_state.fileids = []

if 'reloadindex' not in st.session_state:
    st.session_state.reloadindex=False

if 'sessionid' not in st.session_state:
    #st.session_state.sessionid=random.randint()
    
    st.session_state.sessionid=12345
    
with st.container():
    with st.container():
                
        if input_text:
            with st.chat_message('user'):
                st.markdown(input_text)
            
            st.session_state.chathistory.append({'role':'user', 'text': input_text})
            
            chat_response = generate_response(input_text, 1)
            st.session_state.reloadindex=False
            with st.chat_message('assistant'):
                st.markdown(chat_response)
                
                #if 'doc_source' in chat_response:
                #    st.markdown('Source Reports:')
                #    st.markdown(chat_response['doc_source'])
            
            st.session_state.chathistory.append({'role':'assistant', 'text': chat_response})
