import requests
from bs4 import BeautifulSoup
import json
import re
import os

# Define the URL
base_url = 'https://www.moneycontrol.com/personal-finance/'

section_types = ['investing', 'insurance', 'banking/', 'financialplanning', 'explainer']

# Adjust the selectors based on the HTML structure of the page
# Assuming each name and description is in a <div> with a specific class

save_path = "C:\\Henry\\Learning\\IICS_CDS_Capstone\\scrapped_data\\financial_articles\\moneycontrol\\"

all_files = os.listdir(save_path)

all_files.extend(os.listdir(save_path + "\\indexed\\"))

def write_to_file(name, article_body):
    filename = name[name.rfind("/") + 1 : name.rfind(".html")] + ".txt"
    
    filename = save_path + filename
    
    with open(filename, "w", encoding="utf-8") as file:
        file.write(article_body)

def scrape_data():
    
    for section_type in section_types:        
    
        # Send a GET request to fetch the raw HTML content
        response = requests.get(base_url + section_type)
        response.raise_for_status()  # Check if the request was successful
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
    
        for entry in soup.find_all('div', class_='news_thumb'):
            links = entry.find_all('a')  # Adjust if the tag or class is different
            for link in links:
                name = link.get('href')
                
                if name.endswith(".html"):
                    
                    filename = name[name.rfind("/") + 1 : name.rfind(".html")] + ".txt"
                    
                    if all_files.count(filename) > 0:
                        print(f"File already loaded: {name[name.rfind('/') + 1 : ]}")
                        continue
                    
                    print('https://www.moneycontrol.com' + name)
                    
                    # Send a GET request to fetch the raw HTML content
                    sub_response = requests.get('https://www.moneycontrol.com' + name)
                    sub_response.raise_for_status()
                    
                    content_soup = BeautifulSoup(sub_response.text, 'html.parser')
                    
                    script_tags = content_soup.find_all('script', type='application/ld+json')
                    
                    for script_tag in script_tags:                
                        if script_tag and script_tag.text.count("\"@type\": \"NewsArticle\"") == 1:                    
                            #print(script_tag.text)                        
                            
                            cleaned_json_string = re.sub(r'[\x00-\x1F\x7F]', '', script_tag.text)
                            
                            json_data = json.loads(cleaned_json_string.strip())
                            
                            # Step 4: Extract the 'articleBody' from the JSON data
                            #entry = json_data[0]
                            if isinstance(json_data, dict):
                                if json_data["@type"] == "NewsArticle":
                                    article_body = json_data.get("articleBody", "No article body found")
                                    write_to_file(name, article_body)
                            elif isinstance(json_data, list):                        
                                for entry in json_data:
                                    if entry["@type"] == "NewsArticle":
                                        article_body = entry.get("articleBody", "No article body found")
                                        write_to_file(name, article_body)


scrape_data()