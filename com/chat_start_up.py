import os
import subprocess
import logging
from logging.handlers import RotatingFileHandler
import sys

def test_non_ui():
    file_path = os.path.join(os.path.dirname(__file__), "../", "wise_invest.log")
    
    handler = RotatingFileHandler(file_path, maxBytes=5000000, backupCount=5, encoding='utf-8')
    
    logging.basicConfig(handlers=[handler],
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s'
                        )
    
    logger = logging.getLogger(__name__)
    
    logger.info("Starting the app")
    
    from com.iisc.cds.cohort7.grp11.advisor_service_direct_agent import generate_response
    
    while True:
        usrid = input("UserId:")
        query = input("Query:")
        generate_response(query, usrid)

def main():
    
    os.environ["CONFIG_PATH"] = os.path.join(os.path.dirname(__file__), "../", "config.properties")
    
    #logger.info(f"config path: { os.getenv('CONFIG_PATH')}")
    
    query = 'what would have been the value of Rs 100000 invested in reliance industries about an year ago be today'
    query = "i am 40 years old, how much do i have to invest to get pension of Rs 100000 per month after retirement"
    
    file_path = os.path.join(os.path.dirname(__file__), "iisc", "cds", "cohort7", "grp11", "ui", "chat_ui_new.py")
    
    #print(f'File path: {file_path}')
    
    #test_non_ui()
    
    process = subprocess.Popen(["streamlit", "run", file_path])

if __name__ == "__main__":
    main()