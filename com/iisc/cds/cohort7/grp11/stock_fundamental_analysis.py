'''
Created on 11-Nov-2024

@author: henry Martin
'''
import json
import threading
import traceback

import pandas as pd
import yfinance as yf


yahoo_finance_api_lock = threading.Lock()

def perform_fundamental_analysis(ticker):
    
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    
    returns_summary = 'Error getting data. Please try again later'
    try:        
        yahoo_finance_api_lock.acquire()
        print(f'Loading data for {ticker}')
        
        returns_summary = load_stock_data(ticker)
    except:
        traceback.print_exc()
    
    yahoo_finance_api_lock.release()
    return returns_summary

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
    
    #print_stock_details(merged_df, 'nsetick.financials')
    
    df = pd.DataFrame()
    financial_data = {        
        "Net Profit Margin": (nsetick.financials.loc["Net Income"] / nsetick.financials.loc["Total Revenue"])[:3] * 100,
        "Operating Margin": (nsetick.financials.loc["Operating Income"] / nsetick.financials.loc["Total Revenue"])[:3] * 100,
        "Return on Equity": (nsetick.financials.loc["Net Income"] / nsetick.balance_sheet.loc["Stockholders Equity"])[:3] * 100,
        "Return on Assets": (nsetick.financials.loc["Net Income"] / nsetick.balance_sheet.loc["Total Assets"])[:3] * 100,
        "Current Ratio": (nsetick.balance_sheet.loc["Current Assets"] / nsetick.balance_sheet.loc["Current Liabilities"])[:3],
        "Debt-to-Equity Ratio": (nsetick.balance_sheet.loc["Total Debt"] / nsetick.balance_sheet.loc["Stockholders Equity"])[:3],
        "Interest Coverage Ratio": (nsetick.financials.loc["Operating Income"] / nsetick.financials.loc["Interest Expense"])[:3]
    }
        
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
    
    analysis = json.dumps(result, indent=4)
    print(f'Fundamental analysis details for {ticker} is {analysis}')
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

#print(perform_fundamental_analysis("BALAMINES.NS"))