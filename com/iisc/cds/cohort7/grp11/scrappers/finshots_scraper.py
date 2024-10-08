import requests
from bs4 import BeautifulSoup
import os

# Define the URL
base_url = 'https://finshots.in/tag/money'

# Adjust the selectors based on the HTML structure of the page
# Assuming each name and description is in a <div> with a specific class

save_path = "C:\\Henry\\Learning\\IICS_CDS_Capstone\\scrapped_data\\financial_articles\\finshots\\"

all_files = os.listdir(save_path)
all_files.extend(os.listdir(save_path + "\\indexed\\"))

def write_to_file(name, article_body):
    filename = name[name.rfind("/") + 1 : name.rfind(".html")] + ".txt"
    
    filename = save_path + filename
    
    with open(filename, "w", encoding="utf-8") as file:
        file.write(article_body)

def scrape_data():
    
    # Send a GET request to fetch the raw HTML content
    response = requests.get(base_url)
    response.raise_for_status()  # Check if the request was successful
    
    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    for link in soup.find_all('a', class_='post-card-image-link'):
        name = link.get('href')
        
        filename = name[10 : name.rfind("/")] + ".txt"
        
        if all_files.count(filename) > 0:
            print(f"File already loaded: {name[name.rfind('/') + 1 : ]}")
            continue
        
        print(base_url + name)
        
        # Send a GET request to fetch the raw HTML content
        sub_response = requests.get("https://finshots.in" + name)
        sub_response.raise_for_status()
        
        content_soup = BeautifulSoup(sub_response.text, 'html.parser')
        
        div_tag = content_soup.find('div', class_='post-content')
        
        print(div_tag.text)                        
        
        write_to_file(name[10 : name.rfind("/")], div_tag.text)
        

scrape_data()