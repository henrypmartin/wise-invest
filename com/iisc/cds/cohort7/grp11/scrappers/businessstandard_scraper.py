import requests
from bs4 import BeautifulSoup
import json
import re
import os

# Define the URL
base_url = 'https://www.business-standard.com/finance/personal-finance'

# Adjust the selectors based on the HTML structure of the page
# Assuming each name and description is in a <div> with a specific class

save_path = "C:\\Henry\\Learning\\IICS_CDS_Capstone\\scrapped_data\\financial_articles\\businessstd\\"

all_files = os.listdir(save_path)
all_files.extend(os.listdir(save_path + "\\indexed\\"))

def write_to_file(name, article_body):
    filename = name[name.rfind("/") + 1 : name.rfind(".html")] + ".txt"
    
    filename = save_path + filename
    
    with open(filename, "w", encoding="utf-8") as file:
        file.write(article_body)

def scrape_data():
    
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36',
    'Referer': 'https://www.google.com',
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }
    
    # Send a GET request to fetch the raw HTML content
    response = requests.get(base_url, headers=headers)
    response.raise_for_status()  # Check if the request was successful
    
    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    
    script_tags = soup.find_all('script', type='application/ld+json')

    urls = None
    for script_tag in script_tags:
        if script_tag and script_tag.text.count("\"@type\":\"ItemList\"") == 1:
            json_data = json.loads(script_tag.text.strip())
            
            # Extract the URLs
            urls = [item['url'] for item in json_data['itemListElement']]
            

    for url in urls:
        name = url[url.rfind("/") + 1 : url.rfind(".html")]
        
        filename = name + ".txt"
        
        if all_files.count(filename) > 0:
            print(f"File already loaded: {name[name.rfind('/') + 1 : ]}")
            continue
        
        print(base_url + name)
        
        # Send a GET request to fetch the raw HTML content
        sub_response = requests.get(url, headers=headers)
        sub_response.raise_for_status()
        
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