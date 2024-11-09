'''
Created on 28-Sep-2024

@author: Henry
'''

import requests
from datetime import datetime, timedelta

url = "https://www.samco.in/bse_nse_mcx/getBhavcopy"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
    'Referer': 'https://www.samco.in/bhavcopy-nse-bse-mcx',
    'Accept': 'application/zip',
    'Content-Type': 'application/x-www-form-urlencoded'
}

download_path='C:\\Henry\\Learning\\IICS_CDS_Capstone\\scrapped_data\\bhavcopies\\'

def download_bhavcopies(start_date, end_date):
    
    data1 = {
        "start_date": start_date,
        "end_date": end_date,
        'bhavcopy_data[]': ["NSE"],
        "show_or_down": "2"
    }
    
    response = requests.post(url, headers=headers, data=data1)
    
    if response.status_code == 200:
        
        filename = start_date + ".zip"  # Replace with your desired filename
        with open(download_path + filename, "wb") as f:
            f.write(response.content)
            f.flush()
    else:
        print("Request failed with status code:", response.status_code)

# Function to get the first and last day of a given month and year
def get_month_date_range(year, month):
    start_date = datetime(year, month, 1)
    # Calculate last day of the month
    end_date = start_date + timedelta(days=31)  # Go to the next month
    end_date = end_date.replace(day=1) - timedelta(days=1)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


end_year = datetime.now().year
end_month = datetime.now().month
end_date = datetime.now().day

start_year = 2024
start_month = 10

while start_year <= end_year:
    
    while (start_year < end_year and start_month <= 12) or (start_year == end_year and start_month <= end_month):
        start_date, end_date = get_month_date_range(start_year, start_month)
        
        print(f'Startdate: {start_date}, enddate: {end_date}')
        download_bhavcopies(start_date, end_date)
        
        start_month += 1        
        
    start_year += 1   
    
    if start_month > 12:
        start_month = 1
    

