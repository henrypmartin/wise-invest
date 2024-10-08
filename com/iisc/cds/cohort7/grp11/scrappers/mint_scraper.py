import requests
from bs4 import BeautifulSoup
import json
import re
import os

# Define the URL
base_url = 'https://www.livemint.com/topic/financial-planning/page-'

section_types = ['investing', 'insurance', 'banking/', 'financialplanning', 'explainer']

# Adjust the selectors based on the HTML structure of the page
# Assuming each name and description is in a <div> with a specific class

save_path = "C:\\Henry\\Learning\\IICS_CDS_Capstone\\scrapped_data\\financial_articles\\mint\\"

all_files = os.listdir(save_path)
all_files.extend(os.listdir(save_path + "\\indexed\\"))

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36',
    'Referer': 'https://www.google.com',
    'Accept-Language': 'en-US,en;q=0.9',
    }
    
def write_to_file(name, article_body):
    filename = name[name.rfind("/") + 1 : name.rfind(".html")] + ".txt"
    
    filename = save_path + filename
    
    with open(filename, "w", encoding="utf-8") as file:
        file.write(article_body)

def scrape_data():
    
    response = requests.get('https://www.livemint.com/topic/financial-planning/', headers=headers)
    response.raise_for_status()  # Check if the request was successful
    
    parse_response_text(response)
    
    for page_no in range(1, 100):        
    
        print(f"Page: {page_no}")
        # Send a GET request to fetch the raw HTML content
        response = requests.get(base_url + str(page_no), headers=headers)
        response.raise_for_status()  # Check if the request was successful
        
        parse_response_text(response)

def parse_response_text(response):
    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    for entry in soup.find_all('div', class_='topicsPage_listingStory__7_As0'):
        links = entry.find_all('a')  # Adjust if the tag or class is different
        for link in links:
            name = link.get('href')
            
            if name and name.endswith(".html"):
                
                filename = name[name.rfind("/") + 1 : name.rfind(".html")] + ".txt"
                
                if all_files.count(filename) > 0:
                    print(f"File already loaded: {name[name.rfind('/') + 1 : ]}")
                    continue
                
                print(name)
                
                # Send a GET request to fetch the raw HTML content
                sub_response = requests.get(name, headers=headers)
                
                try:
                    sub_response.raise_for_status()
                except:
                    try:
                        sub_response = requests.get(name.replace('money/personal-finance', 'topic/financial-planning'), headers=headers)
                        sub_response.raise_for_status()
                    except:
                        continue
                
                content_soup = BeautifulSoup(sub_response.text, 'html.parser')
                
                script_tags = content_soup.find_all('script', type='application/ld+json')
                
                for script_tag in script_tags:                
                    if script_tag and script_tag.text.count("\"@type\":\"NewsArticle\"") == 1:                    
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