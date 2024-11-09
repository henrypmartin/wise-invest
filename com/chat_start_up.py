import os
import subprocess

def main():
    
    os.environ["CONFIG_PATH"] = os.path.join(os.path.dirname(__file__), "../", "config.properties")
    
    print(f"config path: { os.getenv('CONFIG_PATH')}")
    
    
    query = "i am 40 years old, how much do i have to invest to get pension of Rs 100000 per month after retirement"
    #from com.iisc.cds.cohort7.grp11.advisor_service_openai import generate_response
    #generate_response(query, 1)
    
    file_path = os.path.join(os.path.dirname(__file__), "iisc", "cds", "cohort7", "grp11", "ui", "chat_ui.py")
    
    print(f'File path: {file_path}')
    
    process = subprocess.Popen(["streamlit", "run", file_path])

if __name__ == "__main__":
    main()