'''
Created on 11-Nov-2024

@author: henry Martin
'''
import json
import threading
import traceback

import pandas as pd
import yfinance as yf
from yahooquery import Ticker
import logging

logger = logging.getLogger(__name__)


yahoo_finance_api_lock = threading.Lock()

import requests

def get_ticker(mf_name):    
    url = "https://query2.finance.yahoo.com/v1/finance/search"
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
    
    def get_mf_symbol(mf_name_updated):
        company_code = None
        params = {"q": mf_name_updated, "quotes_count": 1, "country": "India"}
        try:
            logger.info(f'Getting mutual fund symbol for: {mf_name_updated}')
            res = requests.get(url=url, params=params, headers=headers)
            data = res.json()
            logger.info(data['quotes'])
            
            if 'quotes' in data:
                for mf_quote in data['quotes']:
                    if mf_quote['exchange'] == 'BSE' and mf_quote['isYahooFinance'] and mf_quote['longname'].find('IDCW') < 0:
                        company_code = mf_quote['symbol']
            
            return company_code
    
        except:
            logger.error(f"Error occurred while processing request: {traceback.format_exc()}")
        
        return company_code
    
    mf_name = mf_name.lower()
    company_code = get_mf_symbol(mf_name)
    
    #if not company_code:
    #    mf_name = mf_name.replace("smallcap", "small cap").replace("midcap", "mid cap").replace("largecap", "large cap")
    #    company_code = get_mf_symbol(mf_name)
    
    return company_code

#print(get_ticker('sbi small cap fund'))

def perform_mutual_fund_analysis(mf_name: str, ticker: str):
    
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    
    returns_summary = 'Error getting data. Please try again later'
    try:        
        yahoo_finance_api_lock.acquire()
                
        ticker_for_name = get_ticker(mf_name)
        
        if ticker_for_name and ticker_for_name != ticker:
            logger.info(f'Replacing MF ticker {ticker} with {ticker_for_name} for mutual fund {mf_name}')
            ticker = ticker_for_name
        
        logger.info(f'Loading data for MF {mf_name} : {ticker}')
        
        returns_summary = load_mf_data(ticker)
    except:
        logger.error(f"Error occurred while processing MF request: {traceback.format_exc()}")
    
    yahoo_finance_api_lock.release()
    return returns_summary

def print_mf_details(data_type, data):
    print(f'*******************{data_type}************************')
    print(data)
    #pass
  
def load_mf_data(ticker):

    nsetick = yf.Ticker(ticker)
    
    fund_details = {}
    
    nav = nsetick.history(period="1mo")
    nav = nav.reset_index()
    nav["Date"] = pd.to_datetime(nav["Date"]).dt.strftime("%Y-%m-%d")
    
    nav = nav[['Date', 'High']]
    nav = nav.iloc[-1]
    logger.info(f'Nav for MF {nav["Date"]} : {nav["High"]}')
    
    fund_details[f'NAV as of {nav["Date"]}'] = nav["High"]
    
    asset_classes = {key: value * 100 for key, value in nsetick.funds_data.asset_classes.items() if value != 0}
        
    fund_details['Asset class wise allocations'] = asset_classes
    
    print_mf_details('asset_classes', asset_classes)
    #print_mf_details('bond_holdings', nsetick.funds_data.bond_holdings)
    
    bond_ratings = {key: value for key, value in nsetick.funds_data.bond_ratings.items() if value != 0}
    print_mf_details('bond_ratings', bond_ratings)
    
    if len(bond_ratings) > 0:
        fund_details['Ratings wise bonds count'] = bond_ratings
    
    #print_mf_details('description', nsetick.funds_data.description)
    #print_mf_details('equity_holdings', nsetick.funds_data.equity_holdings)
    #print_mf_details('fund_operations', nsetick.funds_data.fund_operations)
    #print_mf_details('fund_overview', nsetick.funds_data.fund_overview)
    
    sector_weightings = {key: value * 100 for key, value in nsetick.funds_data.sector_weightings.items() if value != 0}
    print_mf_details('sector_weightings', sector_weightings)
    
    if len(sector_weightings) > 0:
        fund_details['Sector Weightings'] = sector_weightings
    
    top_holdings = nsetick.funds_data.top_holdings
        
    if not top_holdings.empty:
        print_mf_details('top_holdings', top_holdings)
        fund_details['Top Holdings'] = top_holdings.to_dict()
    
    
    #print('-----------------------------------------------------------------------------------------------------------------------------------')
    #print('-----------------------------------------------------------------------------------------------------------------------------------')
    nsetick = Ticker(ticker)
    
    #print_mf_details('financial_data', nsetick.financial_data)
    #print_mf_details('fund_bond_holdings', nsetick.fund_bond_holdings)
    #print_mf_details('fund_category_holdings', nsetick.fund_category_holdings)
    #print_mf_details('fund_equity_holdings', nsetick.fund_equity_holdings)
    #print_mf_details('fund_holding_info', nsetick.fund_holding_info)
    #print_mf_details('fund_ownership', nsetick.fund_ownership)
    
    #fund_performance = {}
    
    def check_and_add_fund_details(input_key, data_dict, output_key, is_percent=True):
        
        multiplier = 100
        
        if not is_percent:
            multiplier = 1
        
        if input_key in data_dict:
            fund_details[output_key] = data_dict[input_key] * multiplier
        
    
    performanceOverview = nsetick.fund_performance[ticker]['performanceOverview']
    check_and_add_fund_details('ytdReturnPct', performanceOverview, 'Year to date %')
    check_and_add_fund_details('oneYearTotalReturn', performanceOverview, '1 Year %')
    check_and_add_fund_details('threeYearTotalReturn', performanceOverview, '3 Years %')
    check_and_add_fund_details('fiveYrAvgReturnPct', performanceOverview, '5 Years %')
    
    annualTotalReturns = nsetick.fund_performance[ticker]['annualTotalReturns']['returns']
    
    for annual_rets in annualTotalReturns:
        check_and_add_fund_details('annualValue', annual_rets, f'Returns for year: {annual_rets["year"]}')

    rankInCategory = nsetick.fund_performance[ticker]['rankInCategory']
    check_and_add_fund_details('ytd', rankInCategory, 'Year to date rank', False)
    check_and_add_fund_details('oneYear', rankInCategory, '1 Year rank', False)
    check_and_add_fund_details('threeYear', rankInCategory, '3 Years rank', False)
    check_and_add_fund_details('fiveYear', rankInCategory, '5 Years rank', False)
    
    riskOverviewStatistics = nsetick.fund_performance[ticker]['riskOverviewStatistics']
    fund_details['Risk Rating'] = riskOverviewStatistics['riskRating']
    
    for risk_statistics in riskOverviewStatistics['riskStatistics']:
        if 'alpha' in risk_statistics:
            fund_details[f'{risk_statistics["year"]} alpha'] = risk_statistics['alpha']
    
    if 'fundProfile' in nsetick.all_modules[ticker]:
        check_and_add_fund_details('family', nsetick.all_modules[ticker]['fundProfile'], 'Fund Family', False)
    
    if 'summaryDetail' in nsetick.all_modules[ticker]:
        check_and_add_fund_details('fiftyTwoWeekLow', nsetick.all_modules[ticker]['summaryDetail'], '52-week low', False)
        check_and_add_fund_details('fiftyTwoWeekHigh', nsetick.all_modules[ticker]['summaryDetail'], '52-week high', False)
    
    if 'esgScores' in nsetick.all_modules[ticker]:
        check_and_add_fund_details('totalEsg', nsetick.all_modules[ticker]['esgScores'], 'ESG Score', False)
        check_and_add_fund_details('ratingYear', nsetick.all_modules[ticker]['esgScores'], 'ESG rating year', False)
        check_and_add_fund_details('ratingMonth', nsetick.all_modules[ticker]['esgScores'], 'ESG rating month', False)
    
    if 'defaultKeyStatistics' in nsetick.all_modules[ticker]:        
        check_and_add_fund_details('morningStarOverallRating', nsetick.all_modules[ticker]['defaultKeyStatistics'], 'Morningstar Overall Rating', False)
        check_and_add_fund_details('morningStarRiskRating', nsetick.all_modules[ticker]['defaultKeyStatistics'], 'Morningstar Risk Rating', False)
        check_and_add_fund_details('fundInceptionDate', nsetick.all_modules[ticker]['defaultKeyStatistics'], 'Fund Inception', False)
    
    analysis = json.dumps(fund_details, indent=4)
    
    #print(f'MF analysis for {ticker} is {analysis}')
    
    return analysis
    
#print(perform_mutual_fund_analysis("SBI small cap fund", "xxxxxx.BO"))
#print(perform_mutual_fund_analysis("0P0001BAGX.BO"))
#print(get_ticker("0P0000KV39.BO"))
