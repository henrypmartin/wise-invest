from langchain_google_community import GoogleSearchAPIWrapper
import os

os.environ["GOOGLE_CSE_ID"] = "942906d903fb84b76"
os.environ["GOOGLE_API_KEY"] = "AIzaSyBkJfPSFkVcoUM7hAVk2awB_8wWIJslWF4"

search = GoogleSearchAPIWrapper()

results = search.results("suggest best large cap mutual funds in India. Only show most recent data from last 30 days", num_results=10)

#print(type(results))
#print(dir(search))

for srch_res in results:        
    print(srch_res['link'])
    print(srch_res['snippet'])
    print('***********************************************************************')
    
