'''
Created on 03-Nov-2024

@author: Henry Martin
'''

import os
import shutil
import pandas as pd
import yfinance as yf

bhavcopies = "C:\\Henry\\Learning\\IICS_CDS_Capstone\\scrapped_data\\bhavcopies"

extract_dir = "C:\\Henry\\Learning\\IICS_CDS_Capstone\\scrapped_data\\bhavcopies\\unzipped"

consolidated_file = "C:\\Henry\\Learning\\IICS_CDS_Capstone\\scrapped_data\\bhavcopies\\consolidated\\daily_stocks_data.txt"

stocks_data_file = "C:\\Henry\\Learning\\IICS_CDS_Capstone\\scrapped_data\\bhavcopies\\consolidated\\stocks_data.txt"

zip_files = os.listdir(bhavcopies)

stock_symbol_to_name = {}

isin_to_stock_symbol = {}

stock_symbol_replacement = {}

for zipfile in zip_files:
    if zipfile.endswith('.zip'):        
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        os.mkdir(extract_dir)
        print(f'Unzipping {zipfile}')
        src_file = os.path.join(bhavcopies, zipfile)
        shutil.unpack_archive(src_file, extract_dir)
        
        all_files = os.listdir(extract_dir)
        
        for bhavcopy in all_files:
            daily_file = os.path.join(extract_dir, bhavcopy)
            daily_df = pd.read_csv(daily_file)
            daily_df = daily_df.drop(columns=['SERIES', 'OPEN', 'HIGH', 'LOW', 'LAST', 'PREVCLOSE', 'TOTTRDQTY', 'TOTTRDVAL','TOTALTRADES'])
            
            for symb, isin in zip(daily_df['SYMBOL'], daily_df['ISIN']):
                if isin in isin_to_stock_symbol:
                    exit_symb = isin_to_stock_symbol[isin]
                    
                    if exit_symb != symb:
                        stock_symbol_replacement[exit_symb] = symb
                else:
                    isin_to_stock_symbol[isin] = symb

        shutil.rmtree(extract_dir)

print(stock_symbol_replacement)