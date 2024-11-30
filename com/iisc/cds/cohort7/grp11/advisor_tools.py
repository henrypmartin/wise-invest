'''
Created on 11-Nov-2024

@author: Henry Martin
'''
import traceback
import requests
import asyncio

from langchain_community.tools.tavily_search import TavilySearchResults

from com.iisc.cds.cohort7.grp11.stock_investment_analysis import perform_investment_analysis
from com.iisc.cds.cohort7.grp11.stock_fundamental_analysis import perform_fundamental_analysis
from com.iisc.cds.cohort7.grp11.mf_analysis import perform_mutual_fund_analysis
from com.iisc.cds.cohort7.grp11.portfolio_allocator import run_portfolio_allocator_details
import logging

logger = logging.getLogger(__name__)

search = TavilySearchResults(max_results=10, include_raw_content=False, include_images=False)

def get_ticker(company_name):    
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
    params = {"q": company_name, "quotes_count": 1, "country": "India"}

    company_code = company_name.upper()
    try:
        res = requests.get(url=url, params=params, headers=headers)
        data = res.json()
        logger.info(f'Get ticket details: {data}')
        if len(data['quotes']) > 0:
            for stk_quote in data['quotes']:
                if stk_quote['exchange'] in ['NSI', 'BSE'] and stk_quote['isYahooFinance'] and stk_quote['longname'].find('IDCW') < 0:
                    company_code = stk_quote['symbol']
                    break
    except:
        logger.error(f"Error occurred while processing request: {traceback.format_exc()}")
        
    return company_code

def process_specific_mutual_fund_queries(mf_name: str, mf_ticker: str) -> str:
    '''  Tool to answer any queries related to specific mutual funds. This tool will be used when the query has mutual fund name in it.
        The output from process_generic_mutual_fund_queries with mutual fund names as expected by Yahoo finance API can be passed as input to this tool
        Args:
            mf_name: the mutual fund name as expected by Yahoo finance API
            mf_ticker: the mutual fund ticker symbol in the form as expected by Yahoo finance API eg xxxxxxxxxx.BO
            '''
    
    logger.info(f'process_specific_mutual_fund_queries args: {mf_name}; {mf_ticker}')
    
    search.include_domains = ["www.valueresearchonline.com", "www.paytmmoney.com", "www.moneycontrol.com"]
    
    invoke_op = search.invoke(mf_name  + 
                              f". Assets under management (AUM) or scheme asset size for {mf_name}. Fund Managers for {mf_name}." +
                              f"Expense Ratio for {mf_name}")
    
    analysis = perform_mutual_fund_analysis(mf_name, mf_ticker)
    
    search_data = [analysis]
    for op in invoke_op:
        search_data.append(op['content'])
        logger.info(f"{op['url']} == {op['content']}")
    
    return ' '.join(search_data) 

def process_generic_mutual_fund_queries(user_query: str) -> str:
    '''  Tool to answer any generic queries related to mutual funds when user query does not have any specific mutual fund name in it.
        Also make calls to process_specific_mutual_fund_queries to get most accurate results
        Args:
            user_query: question on mutual funds including the mutual fund name as understood by Yahoo finance
            '''
    
    logger.info(f'process_generic_mutual_fund_queries args: {user_query}')
    
    search.include_domains = ["www.valueresearchonline.com", "www.paytmmoney.com", "www.moneycontrol.com"]
    
    invoke_op = search.invoke(user_query)
    
    #analysis = perform_mutual_fund_analysis(mf_name, mf_ticker)
    
    #search_data = [analysis]
    search_data = []
    for op in invoke_op:
        search_data.append(op['content'])
        logger.info(f"{op['url']} == {op['content']}")        
    
    return ' '.join(search_data) 
    
def process_generic_queries(user_query: str) -> str:
    ''' Returns responses to generic user queries based on the web search.
        Make calls to any of the below tools as appropriate:
            process_stock_fundamentals_queries: to get fundamental details of companies
            calculate_stock_investment_returns: to get historic returns analysis of companies
    '''
    
    #search.include_domains = ["morningstar.in"]
    
    logger.info(f'process_generic_queries args: {user_query} ')
    
    invoke_op = search.invoke(user_query)
    search_data = []
    for op in invoke_op:
        search_data.append(op['content'])
        logger.info(f"{op['url']} == {op['content']}")
        logger.info('7777777777777777777777777777777777777777777')
    
    return ' '.join(search_data)     

def calculate_stock_investment_returns(company_name: str, ticker: str, invested_date: str, invested_amount: int=100000) -> str:
    ''' Tool to answer any queries related to current value of invested amount in stocks and investment date in the past.
        Args:
            company_name: company name of listed Indian company
            ticker: the ticker symbol for listed Indian company in the form as expected by Yahoo finance API eg INFY.NS for Infosys limited
            invested_date: invested date in YYYY-mm-dd format 
            invested_amount: invested amount.'''
    
    logger.info(f'calculate_investment_returns args: {company_name}; {ticker}; {invested_date}; {invested_amount}')
    
    ticker = get_ticker(company_name)
    logger.info(f'Ticker from YahooFinance API: {ticker}')
    data = perform_investment_analysis(ticker, invested_date, invested_amount)
    
    return data

def process_stock_fundamentals_queries(user_query: str, ticker: str) -> str:
    ''' Tool to answer queries related to any fundamentals of listed companies in India.
        Args:
            user_query: question including the company name as understood by 5paisa.com
            ticker: the ticker symbol for listed Indian company in the form as expected by Yahoo finance API eg INFY.NS for Infosys limited"
     '''
    
    logger.info(f'process_stock_fundamentals_queries args: {user_query}; {ticker}')
    
    search.include_domains = ["www.5paisa.com"]
    
    invoke_op = search.invoke(user_query)
    logger.info(f'process_company_queries output: {invoke_op}')
    
    analysis = perform_fundamental_analysis(ticker)
    
    search_data = [analysis]
    for op in invoke_op:
        search_data.append(op['content'])
        logger.info(f"{op['url']} == {op['content']}")        
    
    return ' '.join(search_data) 

def process_company_queries(user_query: str, company_name: str, ticker: str) -> str:
    ''' Tool to answer queries related to any listed companies in India.
        If required, make calls to process_stock_fundamentals_queries to get details of other companies
        Args:
            user_query: question including the company name as understood by 5paisa.com
            company_name: company name of listed Indian company
            ticker: the ticker symbol for listed Indian company in the form as expected by Yahoo finance API"
     '''
    
    logger.info(f'process_company_queries args: {user_query}; {company_name}; {ticker}')
    
    search.include_domains = ["www.5paisa.com"]
    
    invoke_op = search.invoke(user_query)
    logger.info(f'process_company_queries output: {invoke_op}')
    
    ticker = get_ticker(company_name)
    logger.info(f'Ticker from YahooFinance API: {ticker}')
    analysis = perform_fundamental_analysis(ticker)
    
    search_data = [analysis]
    for op in invoke_op:
        search_data.append(op['content'])
        logger.info(f"{op['url']} == {op['content']}")        
    
    return ' '.join(search_data) 

def get_historic_stock_data(user_input: str, period:str) -> str:
    ''' Returns historic stock data for given ticker and period.
    Eg for period is 1Y, 2Y, 5Y etc or could be date in yyyy-mm-dd format'''
    
    logger.info(f'get_historic_stock_data args: {user_input} and {period}')
    
    #search.include_domains = [f"https://www.google.com"]
    #data_df = yf.download(ticker, period=period)
    #output = search.invoke(f'{ticker} and {period}')
    output = search.invoke(f'{user_input} stock price {period}')
    
    logger.info(output)
    
    #print(data_df)
    #print(type(data_df))
    
    return f'Input details are {user_input} and {user_input}' 

def run_portfolio_allocator(query: str, risk_profile: str, invested_amount: float) -> str:
    ''' Tool to answer any queries related to portfolio allocation or suggestions on investment allocation.
        Args:
            query: user query
            risk_profile: risk profile of user eg aggressive, low, moderate etc
            invested_amount: amount to be invested .'''
    
    logger.info(f'run_portfolio_allocator args: {query}; {risk_profile}; {invested_amount}')
    
    data = asyncio.run(run_portfolio_allocator_details(query, risk_profile, invested_amount))
    
    return data