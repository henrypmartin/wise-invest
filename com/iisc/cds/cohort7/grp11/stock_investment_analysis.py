'''
Created on 11-Nov-2024

@author: henry Martin
'''
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import traceback
import threading
import logging

logger = logging.getLogger(__name__)

yahoo_finance_api_lock = threading.Lock()

def perform_investment_analysis(ticker, start_date, investment_amount):
    
    returns_summary = 'Error getting data. Please try again later'
    try:        
        yahoo_finance_api_lock.acquire()
        logger.info(f'Loading data for {ticker}')
        
        curr_price_data = yf.download(ticker, period='1d')
        
        curr_share_price = curr_price_data.Low.iloc[0]
        
        logger.info(f'Current price for {ticker} is {curr_share_price}')
        price_data, stk_splits, divs = load_stock_data(ticker, start_date)
        
        logger.info(f'{ticker}: Price Data: \n{price_data}')
        logger.info(f'{ticker}: Stock splits: \n{stk_splits}')
        logger.info(f'{ticker}: Dividends: \n{divs}')
        
        if price_data is None:
            return -1, -1, -1, -1
        
        divs['Date'] = divs['Date'].dt.tz_localize(None)
        price_data['Date'] = price_data['Date'].dt.tz_localize(None)
        stk_splits['Date'] = stk_splits['Date'].dt.tz_localize(None)
        
        logger.info("getting original purchase shares")
        orig_purchase_price, orig_purchase_shares, split_or_bonus_shares = get_initial_shares_count(investment_amount, stk_splits, price_data)
        
        logger.info(f'{ticker}: Original purchase shares count: {orig_purchase_shares}')
        logger.info(f'{ticker}: Adjusted price: \n{price_data.High.iloc[0]}')
        logger.info(f'{ticker}: Split or Bonus Shares?: \n{split_or_bonus_shares}')
        
        logger.info("Adjust for splits and bonus")
        get_split_and_bonus_adjusted_shares_count(stk_splits, orig_purchase_shares)
        
        logger.info(f'{ticker}: Share count: \n{stk_splits.head()}')
            
        if stk_splits.empty:
            current_shares_count = orig_purchase_shares
        else:
            current_shares_count = stk_splits['shares_count'].tail(1).values[0]
        
        logger.info("Get total dividends")
        total_dividend_received = get_total_dividend_amt(divs, current_shares_count)
        logger.info(f'{ticker}: total dividend received: \n{total_dividend_received}')
        
        #Current market value without reinvesting dividend
        mkt_val_wo_div_reinv = curr_share_price * current_shares_count
        mkt_val_wo_div_reinv = np.round(mkt_val_wo_div_reinv.values[0], 2)
        #print(type(mkt_val_wo_div_reinv))
        #print(dir(mkt_val_wo_div_reinv))
        
        logger.info("Get dividend reinvested shares")
        div_reinvest_stks_cnt = get_dividend_reinvested_shares(ticker, price_data, divs, stk_splits)
        
        #Current market value after reinvesting dividend
        mkt_val_wt_div_reinv = curr_share_price * (current_shares_count + div_reinvest_stks_cnt)
        
        mkt_val_wt_div_reinv = np.round(mkt_val_wt_div_reinv, 2)
        
        cagr = calculate_cagr(investment_amount, mkt_val_wo_div_reinv, start_date)
        cagr_div_reinv = calculate_cagr(investment_amount, mkt_val_wt_div_reinv, start_date)
        
        ca_adj_purchase_price_details = ''
        if split_or_bonus_shares:
            ca_adj_purchase_price_details = f'(split/bonus adjusted purchase price: {np.round(price_data.High.iloc[0].values[0], 2)} )' 
        
        orig_purchase_price = np.round(orig_purchase_price.values[0], 2)
        curr_share_price = np.round(curr_share_price.values[0], 2)
        
        if div_reinvest_stks_cnt > 0 :
            returns_summary = f'''{investment_amount} invested in {ticker} would have been 
                {mkt_val_wo_div_reinv}(if dividend not reinvested, current no. of shares {current_shares_count}) and 
                {mkt_val_wt_div_reinv} (if dividend reinvested, current no. of shares {current_shares_count}).
                Total dividend amount received {total_dividend_received}.  
                CAGR returns of {cagr} (if dividend not reinvested) and
                {cagr_div_reinv} (if dividend reinvested).
                Purchase Price: {orig_purchase_price} {ca_adj_purchase_price_details}
                Current Price: {curr_share_price}'''
        else:
            returns_summary = f'''{investment_amount} invested in {ticker} would have been 
                {mkt_val_wo_div_reinv}(current no. of shares {current_shares_count}).
                Total dividend amount received {total_dividend_received}.  
                CAGR returns of {cagr}.
                Purchase Price: {orig_purchase_price} {ca_adj_purchase_price_details}
                Current Price: {curr_share_price}'''
        
        logger.info(returns_summary)
    except:
        logger.error(f"Error occurred while processing request: {traceback.format_exc()}")

    yahoo_finance_api_lock.release()
    return returns_summary


def load_stock_data(ticker, start_date):

    end_date = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=5)
    price_data = yf.download(ticker, start=start_date, end=end_date)
    
    if price_data.empty:
        logger.info("No data available for the given date range.")
        return None, None, None
    
    nsetick = yf.Ticker(ticker)
    
    #Get the stock splits/bonus corp actions
    stk_splits = nsetick.splits
    stk_splits = stk_splits.reset_index()
    
    stk_splits = stk_splits[stk_splits['Date'] > start_date]
    
    stk_splits = stk_splits.reset_index()
    
    #Get dividends
    divs = nsetick.dividends
    
    divs = divs.reset_index()
    
    divs = divs[divs['Date'] > start_date]
    
    divs = divs.reset_index()
    
    price_data = price_data.reset_index()
    
    return price_data, stk_splits, divs


def get_initial_shares_count(investment_amt, stk_splits, price_data):
    total_stk_aplit_ratio = stk_splits['Stock Splits'].product()
    
    ca_adj_purchase_price = price_data.High.iloc[0]
    
    purchase_price = total_stk_aplit_ratio * ca_adj_purchase_price
    
    orig_purchase_shares = int(investment_amt / purchase_price)
    
    split_or_bonus_shares = ca_adj_purchase_price != purchase_price
        
    return purchase_price, orig_purchase_shares, split_or_bonus_shares.values[0]

def get_split_and_bonus_adjusted_shares_count(stk_splits, orig_purchase_shares):

    if stk_splits.empty:
        logger.info("No stock splits/bonus data available.")
        return
    
    # Initialize the new column with NaN values
    stk_splits['shares_count'] = pd.NA
    
    logger.info("getting additional shares count for 1st bonus")
    stk_splits['shares_count'] = stk_splits['Stock Splits'].iloc[0] * orig_purchase_shares
    
    logger.info("getting original purchase shares for other bonus")
    # For the rest of the rows, multiply by the previous value in the new column
    for i in range(1, len(stk_splits)):
        stk_splits.loc[i, 'shares_count'] = stk_splits['shares_count'].iloc[i-1] * stk_splits.loc[i, 'Stock Splits']
        
    
def get_total_dividend_amt(divs, current_shares_count):
    # Initialize the new column with NaN values
    divs['total_div_amt'] = pd.NA
    
    divs['total_div_amt'] = divs['Dividends'] * current_shares_count
    
    total_dividend_received = sum(divs['total_div_amt'])
    
    divs.head(5)
    
    return total_dividend_received

def get_market_value_without_dividend_reinvest(curr_price, stk_splits) :
    market_value_without_div_reinvest = curr_price * stk_splits['shares_count'].tail(1).values[0]
    
    return market_value_without_div_reinvest

def get_dividend_reinvested_shares(ticker, price_data, divs, stk_splits):

    div_reinvest_stks_cnt = 0
    prev_div_date = price_data['Date'].iloc[0]
    
    logger.info(f'{ticker}: Previous div date \n{prev_div_date}')
    
    #divs['Date'] = divs['Date'].dt.tz_localize(None)
    
    for div_row in divs.iterrows():        
        div_dt = div_row[1]['Date']
        div_amt = div_row[1]['total_div_amt']
        stk_price = yf.download(ticker, start=div_dt, end=div_dt.date() + timedelta(days=1))
        
        stk_price = stk_price.reset_index()
        stk_splits_post_divs = stk_splits[stk_splits['Date'] > div_dt]
        
        logger.info(f"Stock split post divs: \n{stk_splits_post_divs}")
        split_ratio = stk_splits_post_divs['Stock Splits'].product()
        
        actual_stk_price = stk_price["High"].iloc[0] * split_ratio
        
        logger.info(f'Adjusted stock price {stk_price["High"].iloc[0]} and actual price {actual_stk_price} on date {stk_price["Date"].iloc[0]}')
        
        div_reinvest_stks_cnt += int(div_amt/actual_stk_price)
        
        stk_splits['Date'] = stk_splits['Date'].dt.tz_localize(None)
        
        #stk_splits_between_divs = stk_splits[stk_splits['Date'] > prev_div_date][stk_splits['Date'] < div_dt]
        stk_splits_between_divs = stk_splits[stk_splits['Date'].between(prev_div_date, div_dt)]
        
        logger.info(f"Stock split between {prev_div_date } and {div_dt}: {stk_splits_between_divs}")
        
        for i in range(0, len(stk_splits_between_divs)):
            div_reinvest_stks_cnt = stk_splits_between_divs['Stock Splits'].iloc[i] * div_reinvest_stks_cnt
            logger.info(f"Applying stock split ratio {stk_splits_between_divs['Stock Splits'].iloc[i]}")
        
        
        prev_div_date = div_dt
        logger.info("*********************************************************")
        #print(f'Stock Price on div date: {div_dt} {div_amt} {stk_price["High"].iloc[0]} {actual_stk_price}')
        
        #div_reinvest_stks_cnt = div_reinvest_stks_cnt + int(div_amt / stk_price["High"].iloc[0])
    
    logger.info(f'Total no of additional stocks after dividend reinvestment: {int(div_reinvest_stks_cnt)}')
    
    return int(div_reinvest_stks_cnt)

def calculate_cagr(beginning_value, ending_value, start_date):

    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.now()

    years = ((end_dt - start_dt).days) / 365.25

    cagr = (ending_value / beginning_value) ** (1 / years) - 1
    return np.round(cagr * 100, 2)

#print(perform_investment_analysis("INFY.NS", "2021-11-16", 100000))