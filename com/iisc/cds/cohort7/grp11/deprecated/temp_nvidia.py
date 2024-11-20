import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.simple import SpaceTokenizer
import html
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
tknzr = SpaceTokenizer()

def data_cleaning_and_preprocessing(text):
    text = html.unescape(text)
    
    txt = tknzr.tokenize(text)
    
    #remove words made up of only special chars 
    txt = [wrd.lstrip().rstrip() for wrd in txt if (not re.fullmatch('[^A-Za-z0-1]+', wrd.lstrip().rstrip())) and len(wrd.lstrip().rstrip()) > 0]
    
    #remove stopwords
    txt = [word for word in txt if word not in stop_words]
    
    #lemmetize
    txt = [lemmatizer.lemmatize(word) for word in txt]
    
    return " ".join(txt)

#file_path = 'C:\\Henry\\Learning\\IICS_CDS_Capstone\\scrapped_data\\financial_articles\\moneycontrol\\3-quality-midcap-stocks-held-by-mutual-funds-12832983.txt'
file_path = 'C:\\Henry\\Learning\\IICS_CDS_Capstone\\scrapped_data\\financial_data\\stocks_data.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    txt = data_cleaning_and_preprocessing(file.read())

print(" ".join(txt))