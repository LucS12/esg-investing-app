#Necessary Packages
import pandas as pd
import numpy as np

#Function to Calculate Annual Returns and Annual Volatility:
def calc_risk_return(data):
    '''
    Parameters
    ----------
    data : Pandas DataFrame
        Will store the annual returns and volatilities calculated.

    Returns
    -------
    Same Pandas DataFrame with annual returns and volatilities.
    '''
    #Read in price data:
    prices_path = r"C:\Users\ldecs\OneDrive\Documents\esgApp\esg\static\data\stock_prices.csv"    
    prices = pd.read_csv(prices_path, index_col='date', parse_dates=True) 
    
    #Calculate daily log returns:
    stock_log_ret = np.log(prices) - np.log(prices.shift(1))
    
    #Calculate and format annual returns into DataFrame:
    data['Return'] = (stock_log_ret.mean() * 252 * 100).round(1)
    
    #Calculate and format annual volatilities into DataFrame:
    data['Volatility'] = (stock_log_ret.std() * np.sqrt(252) * 100).round(1)
    
    return data

#Function to get company metrics and ESG rankings to show:
def get_company(stock):
    '''
    Parameters
    ----------
    stock : STRING
        Stock symbol (Ex: AAPL).

    Returns
    -------
    Dictionary with all ESG score rankings of the stock along with its basic
         metrics/information.
    '''
    #Read in esg score data:
    esg_path = r"C:\Users\ldecs\OneDrive\Documents\esgApp\esg\static\data\stock_data.csv"
    esg_data = pd.read_csv(esg_path, index_col='ticker')
    
    #Get risk and return metrics from function above:
    data = calc_risk_return(esg_data)
    
    #Format market cap data to be in "Millions":
    data['Market_Cap'] = (data['Market_Cap'] / 1000000).round(3)
    
    #Format ESG scores to integer:
    cols = data.columns[:4]
    data[cols] = data[cols].round(1)
    
    #Round Beta to 2 decimals:
    data['Beta_1Y'] = data['Beta_1Y'].round(2)
    
    #Dictionary from all DataFrame data for the chosen stock:
    stock_metrics = data.loc[stock].to_dict()    
    
    #Get Universal ESG score rankings into dictionary:
    total_stocks = len(data)
    keys = ['uni_E', 'uni_S', 'uni_G', 'uni_esg']
    
    for k, col in zip(keys, data.columns[:4]):
        stock_metrics[k] = len(data[col].sort_values(ascending=False).loc[:stock])

    #Filter data to have stocks of the stock sector only:    
    sector = data.loc[stock]['Sectors']
    sector_data = data[data.Sectors == sector]
    
    #Put number of stocks in stock sector into dictionary:
    sec_total = len(sector_data)
    stock_metrics['sec_total'] = sec_total
    
    #Put sector specific ESG score rankings into dictionary: 
    keys2 = ['sec_E', 'sec_S', 'sec_G', 'sec_esg']
    
    for k2, col in zip(keys2, cols):
        stock_metrics[k2] = len(sector_data[col].sort_values(ascending=False).loc[:stock])
        
    return stock_metrics
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    