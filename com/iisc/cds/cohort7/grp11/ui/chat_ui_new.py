import streamlit as st
import os
import logging
from logging.handlers import RotatingFileHandler
import sys
import secrets

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../"))
print(root_path)
sys.path.append(root_path)

file_path = os.path.join(os.path.dirname(__file__), "../../../../../../", "wise_invest.log")

handler = RotatingFileHandler(file_path, maxBytes=5000000, backupCount=5, encoding='utf-8')

logging.basicConfig(handlers=[handler],
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)
    
from com.iisc.cds.cohort7.grp11.advisor_service_direct_agent import generate_response

def load_css(css_file):
    with open(css_file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Add a transparent logo to the sidebar
def add_sidebar_logo():
    iisc_logo_url = "https://iisc.ac.in/wp-content/uploads/2020/08/IISc_Master_Seal_Black_Transparent.png"  # Replace with your transparent logo URL
    ts_logo_url = "https://static.talentsprint.com/ts_drupal/talentsprint/images/logo.webp"
    logo_html = f"""
    <div style="
        display: flex; 
        align-items: center; 
        margin-bottom: 5px;">
        <img src="{ts_logo_url}" alt="TS Logo" style="
            max-width: 50%; 
            height: auto; 
            opacity: 0.8; 
            border-radius: 10px;" />
        <img src="{iisc_logo_url}" alt="IISC Logo" style="
            max-width: 50%; 
            height: auto; 
            opacity: 0.8; 
            border-radius: 10px;" />
    </div>
    """
    #with st.sidebar:
    st.markdown(logo_html, unsafe_allow_html=True)

def main():
    
    # Configure page
    st.set_page_config(
        page_title="WiseInvest",
        page_icon="💰",
        layout="wide"
    )
    
    # Load custom CSS
    file_path = os.path.join(os.path.dirname(__file__), "style.css")
    load_css(file_path) # add your path to stylesheet

    # Sidebar
    with st.sidebar:
        # Load and display IISC logo
        add_sidebar_logo()
        
        st.markdown("## About WiseInvest")
        st.markdown("""
        WiseInvest is your intelligent financial companion, powered by advanced AI 
        to help you make informed investment decisions.
        
        **Key Features:**
        - Personalized investment advice
        - Market trend analysis
        - Risk assessment
        - Portfolio optimization
        - Real-time market insights
        
        Developed by the IISC CDS Course Cohort-7, Group-11 participants.
        """)

    # Main content - centered with max width
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-title">WiseInvest</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Your guide for the financial journey</p>', unsafe_allow_html=True)
    
    # Chatbot   
    # Initialize chat history
    if "chathistory" not in st.session_state:
        st.session_state.chathistory = []

    if "session_id" not in st.session_state:
        st.session_state.session_id = secrets.token_urlsafe(5)
        
    # Display chat messages
    for message in st.session_state.chathistory:
        with st.chat_message(message["role"]):
            st.markdown(message["text"])

    # Chat input
    if input_text := st.chat_input("How can I help you with your investments today?"):
        # Add user message to chat history
        st.session_state.chathistory.append({"role": "user", "text": input_text})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(input_text)

        #chat_response = generate_response(input_text, 1)
        #st.session_state.reloadindex=False
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Fetching response..."):
                chat_response = generate_response(input_text, st.session_state.session_id)
                st.markdown(chat_response)        
                # Add assistant response to chat history
                st.session_state.chathistory.append({"role": "assistant", "text": chat_response})

    
    st.markdown('</div>', unsafe_allow_html=True)  # Close main-content

if __name__ == "__main__":
    main()