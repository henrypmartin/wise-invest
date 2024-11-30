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

def perform_fundamental_analysis(ticker):
    
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    
    returns_summary = 'Error getting data. Please try again later'
    try:        
        yahoo_finance_api_lock.acquire()
        logger.info(f'Loading data for {ticker}')
        
        returns_summary = load_stock_data(ticker)
    except:
        logger.error(f"Error occurred while processing fundamental analysis request: {traceback.format_exc()}")
    
    yahoo_finance_api_lock.release()
    return returns_summary

def add_stock_details(data_desc, data_dict, data_key, result):
    #print(f'*****************************{data_type}******************************************')
    #print(data)
    if data_key in data_dict:
        result[data_desc] = data_dict[data_key]

def load_stock_data(ticker):

    nsetick = yf.Ticker(ticker)
    
    #print_stock_details(nsetick.actions, 'nsetick.actions')
    #print_stock_details(nsetick.analyst_price_targets, 'nsetick.analyst_price_targets')
    balance_sheet = nsetick.balance_sheet
    
    balance_sheet_idx = ['Total Debt', 'Stockholders Equity', 'Retained Earnings', 'Total Assets', 'Cash Cash Equivalents And Short Term Investments', 'Cash And Cash Equivalents']
    balance_sheet_idx = [idx for idx in balance_sheet_idx if idx in balance_sheet.index]
    
    balance_sheet = balance_sheet.loc[balance_sheet_idx]
    balance_sheet = balance_sheet.iloc[:, :3]
    #print_stock_details(balance_sheet, 'nsetick.balance_sheet')
    #print_stock_details(nsetick.basic_info, 'nsetick.basic_info')
    #print_stock_details(nsetick.calendar, 'nsetick.calendar')
    #print_stock_details(nsetick.capital_gains, 'nsetick.capital_gains')
    cash_flow = nsetick.cash_flow
    
    cash_flow_idx = ['Free Cash Flow', 'End Cash Position', 'Beginning Cash Position', 'Cash Flow From Continuing Operating Activities']
    cash_flow_idx = [idx for idx in cash_flow_idx if idx in cash_flow.index]
        
    cash_flow = cash_flow.loc[cash_flow_idx]
    cash_flow = cash_flow.iloc[:, :3]
    #print_stock_details(cash_flow, 'nsetick.cash_flow')
    
    dividends = nsetick.dividends
    dividends = dividends.reset_index()
    dividends["Date"] = pd.to_datetime(dividends["Date"]).dt.strftime("%Y-%m-%d")
    
    #print_stock_details(dividends, 'nsetick.dividends')
    #print_stock_details(nsetick.earnings, 'nsetick.earnings')
    #print_stock_details(nsetick.eps_trend, 'nsetick.eps_trend')
    #print_stock_details(nsetick.fast_info, 'nsetick.fast_info')
    financials = nsetick.financials
    
    financials_idx = ['EBITDA', 'Basic EPS', 'Net Income', 'Operating Income', 'Total Revenue']
    financials_idx = [idx for idx in financials_idx if idx in financials.index]
    
    financials = financials.loc[financials_idx]
    financials = financials.iloc[:, :3]
    
    merged_df = pd.concat([balance_sheet, cash_flow, financials])
    merged_df.columns = merged_df.columns.astype(str)
    
    financials = nsetick.financials
    #financials = financials.reset_index()
    
    #print_stock_details(financials, 'nsetick.financials')
    
    df = pd.DataFrame()
    financial_data = {}
    
    #print(f'Location: {financials.index}')
    #print(f'Location: {"Net Income" in financials.index}')
    
    if "Net Income" in financials.index and "Total Revenue" in financials.index:
        financial_data["Net Profit Margin"] = (financials.loc["Net Income"] / financials.loc["Total Revenue"])[:3] * 100
    
    if "Operating Income" in financials.index and "Total Revenue" in financials.index:
        financial_data["Operating Margin"] = (financials.loc["Operating Income"] / financials.loc["Total Revenue"])[:3] * 100
    
    if "Net Income" in financials.index and "Stockholders Equity" in nsetick.balance_sheet.index:
        financial_data["Return on Equity"] = (financials.loc["Net Income"] / nsetick.balance_sheet.loc["Stockholders Equity"])[:3] * 100
    
    if "Net Income" in financials.index and "Total Assets" in nsetick.balance_sheet.index:
        financial_data["Return on Assets"] = (financials.loc["Net Income"] / nsetick.balance_sheet.loc["Total Assets"])[:3] * 100
    
    if "Current Assets" in financials.index and "Current Liabilities" in nsetick.balance_sheet.index:
        financial_data["Current Ratio"] = (nsetick.balance_sheet.loc["Current Assets"] / nsetick.balance_sheet.loc["Current Liabilities"])[:3]
    
    if "Total Debt" in financials.index and "Stockholders Equity" in nsetick.balance_sheet.index:
        financial_data["Debt-to-Equity Ratio"] = (nsetick.balance_sheet.loc["Total Debt"] / nsetick.balance_sheet.loc["Stockholders Equity"])[:3]
    
    if "Operating Income" in financials.index and "Interest Expense" in financials.index:
        financial_data["Interest Coverage Ratio"] = (financials.loc["Operating Income"] / financials.loc["Interest Expense"])[:3]
            
    for key, value in financial_data.items():
        dic = {}
        for tup in value.items():
            dic[tup[0].strftime("%Y-%m-%d")] = tup[1] 

        tdf = pd.DataFrame(data=dic, index=[key])
        df = pd.concat([df, tdf])
    
    merged_df = pd.concat([merged_df, df])
    result = {
        "MergedData": merged_df.to_dict(),
        "DividendsHistory": dividends.to_dict(orient="records"),
        "Price-to-Earnings (P/E) Ratio": nsetick.info.get("trailingPE"),
        "Price-to-Book (P/B) Ratio": nsetick.info.get("priceToBook")
        }
    
    nsetick = Ticker(ticker)
    
    if 'assetProfile' in nsetick.all_modules[ticker]:        
        add_stock_details('Long Business Summary', nsetick.all_modules[ticker]['assetProfile'], 'longBusinessSummary', result)
        
        print(f"company offiers: {nsetick.all_modules[ticker]['assetProfile']['companyOfficers']}")
        for cmp_officers in nsetick.all_modules[ticker]['assetProfile']['companyOfficers']:
            if cmp_officers['title'].lower().find("ceo") > 0 or cmp_officers['title'].lower().find("chief executive officer") >= 0:
                add_stock_details('CEO', cmp_officers, 'name', result)            
            elif cmp_officers['title'].lower().find("cfo") > 0 or cmp_officers['title'].lower().find("chief financial officer") >= 0:
                add_stock_details('CFO', cmp_officers, 'name', result)
        
    
    if 'price' in nsetick.all_modules[ticker]:        
        add_stock_details('Current Share Price', nsetick.all_modules[ticker]['price'], 'regularMarketPrice', result)
        add_stock_details('Current Share Price Time', nsetick.all_modules[ticker]['price'], 'regularMarketTime', result)
    
    if 'summaryDetail' in nsetick.all_modules[ticker]:        
        add_stock_details('Previous Close', nsetick.all_modules[ticker]['summaryDetail'], 'previousClose', result)
        add_stock_details('52-Week Low', nsetick.all_modules[ticker]['summaryDetail'], 'fiftyTwoWeekLow', result)
        add_stock_details('52-Week High', nsetick.all_modules[ticker]['summaryDetail'], 'fiftyTwoWeekHigh', result)
        add_stock_details('Dividend Yield', nsetick.all_modules[ticker]['summaryDetail'], 'dividendYield', result)
        add_stock_details('Market Cap', nsetick.all_modules[ticker]['summaryDetail'], 'marketCap', result)
    
    if 'esgScores' in nsetick.all_modules[ticker]:
        add_stock_details('ESG Rating', nsetick.all_modules[ticker]['esgScores'], 'totalEsg', result)
        add_stock_details('Environment Score', nsetick.all_modules[ticker]['esgScores'], 'environmentScore', result)
        add_stock_details('Social Score', nsetick.all_modules[ticker]['esgScores'], 'socialScore', result)
        add_stock_details('Governance Score', nsetick.all_modules[ticker]['esgScores'], 'governanceScore', result)
        add_stock_details('ESG Rating Year', nsetick.all_modules[ticker]['esgScores'], 'ratingYear', result)
        add_stock_details('ESG Rating Month', nsetick.all_modules[ticker]['esgScores'], 'ratingMonth', result)
    
    if 'financialData' in nsetick.all_modules[ticker]:        
        add_stock_details('Revenue Per Share', nsetick.all_modules[ticker]['financialData'], 'revenuePerShare', result)
        add_stock_details('Debt To Equity', nsetick.all_modules[ticker]['financialData'], 'debtToEquity', result)
        add_stock_details('Current Ratio', nsetick.all_modules[ticker]['financialData'], 'currentRatio', result)
    #print_stock_details('returnOnAssets', nsetick.all_modules[ticker]['financialData']['returnOnAssets'])
    #print_stock_details('returnOnEquity', nsetick.all_modules[ticker]['financialData']['returnOnEquity'])
    
    analysis = json.dumps(result, indent=4)
    logger.info(f'Fundamental analysis details for {ticker} is {analysis}')
    #print(analysis)
    #print((nsetick.financials.loc["Net Income"] / nsetick.financials.loc["Total Revenue"])[:3])
    return analysis
    #print_stock_details(nsetick.funds_data, 'nsetick.funds_data')
    #print_stock_details(nsetick.growth_estimates, 'nsetick.growth_estimates')
    #print_stock_details(nsetick.incomestmt, 'nsetick.incomestmt')
    #print_stock_details(nsetick.insider_transactions, 'nsetick.insider_transactions')
    #print_stock_details(nsetick.major_holders, 'nsetick.major_holders')
    #print_stock_details(nsetick.institutional_holders, 'nsetick.institutional_holders')
    #print_stock_details(nsetick.mutualfund_holders, 'nsetick.mutualfund_holders')
    #print_stock_details(nsetick.recommendations, 'nsetick.recommendations')
    #print_stock_details(nsetick.recommendations_summary, 'nsetick.recommendations_summary')
    #print_stock_details(nsetick.splits, 'nsetick.splits')

#print(perform_fundamental_analysis("HDFCBANK.NS"))