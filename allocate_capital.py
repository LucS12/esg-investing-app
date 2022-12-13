#Necessary Packages:
import pandas as pd
import numpy as np
import cvxpy as cp
from cvxpy.atoms.affine.wraps import psd_wrap
from statsmodels.stats.correlation_tools import cov_nearest  
import plotly.graph_objects as go

#Black-Litterman Function to estimate returns and covariance matrix:
def black_litterman(prices, data, A):
    '''
    Parameters
    ----------
    prices : Pandas DataFrame 
        Table of stock symbols with daily stock prices for a period of time.
    mkt_cap : Pandas Series
        Single column table with market cap data for each stock symbol.
    A : float
        Risk-aversion parameter ranging from 0 to 5.
        
    Returns
    -------
    Black-Litterman posterior returns and covariance matrix of assets given by
    Atilio Meucci's methodology of the formula and calculation. 2 Pandas DataFrames.

    '''    
    #Calculate log daily returns + Average yearly log returns:
    stock_log_ret = np.log(prices) - np.log(prices.shift(1))
    avg_yearly_ret = stock_log_ret.mean() * 252
    
    #Views vector (Q) as Average Log Returns:
    Q = avg_yearly_ret[avg_yearly_ret>-1]
    data = data.loc[Q.index, :]
    
    stock_log_ret = stock_log_ret.loc[:, Q.index]
    
    #Weights based on Market Cap for Global/Market Portfolio:
    mcap_wgts = (data.Market_Cap / data.Market_Cap.sum()).values #NP array for calculations to come
    
    #Covariance Matrix of log returns (S):
    S = stock_log_ret.cov()
    
    #Ensure there are no NaNs for covariances: 
    S.fillna(S.mean(), inplace=True)          #Fill with each stock means 
    S.fillna(S.mean().mean(), inplace=True)   #Fill with overall mean for those with mean of 0
    
    #Implied Equilibrium Excess Returns Vector (pi = 2A*S*w -> Meucci):
    pi = 2.0*A*(S @ mcap_wgts)
    
    #Link matrix (P) with 1s showing the position of the stock for that view (return prediction):
    P = np.zeros((len(Q), len(Q)))   #Make a matrix with length of stocks and views
    np.fill_diagonal(P, 1)           #Fill matrix's diagonal with 1 for each stock
    
    #Scalar (tau) and uncertainty of views matrix (omega):
        #tau 0 between 1 --> 1 / length of time series by Meucci
        #c default is 1 by Meucci -> constant rep overall confidence in the views return estimator
        #omega = 1/c * P * S * P^T -> Meucci
    tau = 1.0/float(len(prices))
    c = 1.0
    omega = np.dot(np.dot(P, S), P.T) / c 
    
    #BL Excess Return: (Meucci formula)
        # = pi + tau*S*P^T * (tau*P*S*P^T + omega)^-1 * (Q - P*pi)
    r2 = np.linalg.inv(tau*P@S@P.T + omega)
    post_pi = pi + np.dot((tau*S@P.T) @ r2, (Q - P@pi))
    
    #BL Covariance Matrix: (Meucci formula)
        # = (1+tau)*S - tau^2*S*P.T * (tau*P*S*P.T + omega)^-1 * P*S
    c2 = np.linalg.inv(tau*P@S@P.T + omega) 
    post_S = (1.0+tau)*S - np.dot(np.dot(tau**2.0*np.dot(S, P.T), c2), np.dot(P, S))
    
    #Ensure symmetric and strict positive semi-definite matrix:
    sym_S = (post_S + post_S.T) / 2
    semidef_S = cov_nearest(sym_S)
    
    return post_pi, semidef_S

#Markowitz Optimization Function Using Black-Litterman Returns and Covariance
def allocate_capital(E_scr, S_scr, G_scr, A, sectors=None, stocks=None):
    '''
    Parameters
    ----------
    E_scr : INT
        Score (0-100) for user Environmental care, 100 signifying the most care.
    S_scr : INT
        Score (0-100) for user Social care, 100 signifying the most care.
    G_scr : INT
        Score (0-100) for user Governance care, 100 signifying the most care.
    A : FLOAT
        Risk-aversion parameter of user (0-5), 5 avoiding the most risk.
    sectors : LIST of STRINGS
        Names of sectors.
    stocks : LIST of STRINGS
        Symbols of stocks.

    Returns
    -------
    Pandas DataFrame with the allocations, sectors, betas, and returns for each
         stock along with a dictionary carrying the portfolio variance and ESG
         scores (E, S, G, total ESG).
    
    Asset allocation algorithm based on Markowitz and Modern Portfolio Theory.
         Finding mean-variance allocations with allocation constraints and 
         ESG score constraints to exhibit user preference/input.
    '''
    #Read in price data of stocks:
    prices_path = r"C:\Users\ldecs\OneDrive\Documents\esgApp\esg\static\data\stock_prices.csv"    
    prices = pd.read_csv(prices_path, index_col='date', parse_dates=True)  

    #Read in esg score data:
    esg_path = r"C:\Users\ldecs\OneDrive\Documents\esgApp\esg\static\data\stock_data.csv"
    data = pd.read_csv(esg_path, index_col='ticker')
    
    #Filter out user input of UNWANTED sectors:
    if sectors != None:
        data = data.loc[data['Sectors'].isin(sectors) == False].sort_index()
        prices = prices.loc[:, data.index]
    
    #Filter out user input of UNWANTED stocks:
    if stocks != None:
        data = data.loc[data.index.isin(stocks) == False].sort_index()
        prices = prices.loc[:, data.index]
    
    #Get Returns and Covariance matrix from Black-Litterman function:
    ret, cov = black_litterman(prices, data, A)
    data = data.loc[ret.index]
    cov = psd_wrap(cov)                           #Ensure positive semi-definite matrix
    
    #Set variables necessary for optimization:
    allocations = cp.Variable(len(ret))           #Variable to optimize/find
    
    E = data.E_score.values @ allocations         #Environmental score of portfolio
    S = data.S_score.values @ allocations         #Social score   
    G = data.G_score.values @ allocations         #Governance score 
    esg = data.ESG_score.values @ allocations     #Total ESG score 
    
    var = cp.quad_form(allocations, cov)          #Variance/risk variable to be minimized
    
    #Constraints: sum of allocations need to be 1 (1), stocks can only receive
    #a max of 10% each (2), no shorting (3), and minimum ESG scores (4):
    cons = [cp.sum(allocations)==1, allocations<=0.10, 
            allocations>=0, E>=E_scr, S>=S_scr, G>=G_scr]
    
    #Markowitz mean-variance objective function: 
    obj = cp.Minimize(var - A*ret.values@allocations)
    
    #Optimization of allocation variable given constraints and function above:
    prob = cp.Problem(obj, cons)
    prob.solve()
    
    wgts = np.array(allocations.value.round(3))  #Rounding optimized allocations to 3 decimals

    #Put weights + metrics to be shown into a Pandas DataFrame with respective stock:
    allocations_df = pd.DataFrame(wgts, index=data.index, columns=['Allocations'])
    allocations_df['Sectors'] = data.Sectors
    allocations_df['Beta_1Y'] = data.Beta_1Y
    allocations_df['Returns'] = ret
    
    #Put ESG scores and Variance into Pandas DataFrame:
    port_metrics = {'Var': var.value, 'E_scr':E.value, 'S_scr':S.value, 
                    'G_scr':G.value, 'ESG_scr':esg.value}
    
    return allocations_df, port_metrics

#Function to get portfolio graph with allocations and metrics:
def get_portfolio(E_scr, S_scr, G_scr, A, sectors=None, stocks=None):
    '''
    Parameters
    ----------
    E_scr : INT
        Score (0-100) for user Environmental care, 100 signifying the most care.
    S_scr : INT
        Score (0-100) for user Social care, 100 signifying the most care.
    G_scr : INT
        Score (0-100) for user Governance care, 100 signifying the most care.
    A : FLOAT
        Risk-aversion parameter of user (0-5), 5 avoiding the most risk.
    sectors : LIST of STRINGS
        Names of sectors.
    stocks : LIST of STRINGS
        Symbols of stocks.
    
    Returns
    -------
    1 Dictionary with portfolio metrics of risk, return, and ESG. 2 Plotly donut
         charts (1 for stock allocations and 1 for sector allocations).
    '''
    #Get portfolio allocations and metrics from Markowitz allocation function:
    allocations, port_metrics = allocate_capital(E_scr, S_scr, G_scr, A, sectors, stocks)
    
    #Filter out symbols with 0 allocation (not in portfolio):
    allocations = allocations[allocations['Allocations'] > 0]
    
    #Write DataFrame to CSV for download option for users:
    allocations[['Allocations','Sectors']].to_csv(r'C:\Users\ldecs\OneDrive\Documents\esgApp\esg\static\data\portfolio.csv')

    #Empty dictionary to store formatted/calculated portfolio metrics:
    metrics = {}

    #Round ESG scores into empty dictionary:
    for scr in list(port_metrics.keys())[1:]:
        metrics[scr] = int(round(port_metrics[scr], 0))
        
    #Calculate portfolio metrics into dictionary: Yearly Return, Yearly Volatility, and Beta
    metrics['Ret'] = (allocations.Returns@allocations.Allocations * 252 * 100).round(1)   #252 days in a trading year
    metrics['Vol'] = (np.sqrt(port_metrics['Var']) * np.sqrt(252) * 100).round(1)         #Annual Vol = sqrt(var)*sqrt(trading days) 
    metrics['beta'] = (allocations.Allocations@allocations.Beta_1Y).round(2)              #Weighted average of allocations and individual betas

    #Make allocations standardized as percentage form out of 100:
    allocations.loc[:, 'Allocations'] = (allocations['Allocations'] * 100).round(1)   #Rounding to 1 decimal

    #Get total allocations for each Sector (will be second subplot):
    sector_allocs = allocations.groupby('Sectors').Allocations.sum()

    #Make a Plotly donut chart for the stock allocations:
    fig_stocks = go.Figure(data=[go.Pie(labels=allocations.index,
                                        values=allocations.Allocations.values,
                                        hole=0.35, pull=0.08,
                                        hoverinfo='label + percent')])

    fig_stocks.update_traces(textposition='inside', textfont_size=14)


    fig_stocks.update_layout(font_color="white",
                             legend=dict(font=dict(
                                        size=13)),
                             autosize=False,
                             width=475,
                             height=300,
                             modebar_remove=['toImage', 'hoverClosestPie'],
                             paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(0,0,0,0)",
                             margin=dict(l=0,r=100,b=0,t=0))

    #Make a Plotly donut chart for the allocations by sector:
    fig_sectors = go.Figure(data=[go.Pie(labels=sector_allocs.index,
                                        values=sector_allocs.values,
                                        hole=0.7,
                                        hoverinfo='label + percent')])

    fig_sectors.update_traces(textposition='inside', textfont_size=14)

    fig_sectors.update_layout(font_color="white",
                              legend=dict(y=0.5,
                                          font=dict(size=13)),
                              autosize=False,
                              width=400,
                              height=225,
                              modebar_remove=['toImage', 'hoverClosestPie'],
                              paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              margin=dict(l=0,r=0,b=0,t=0))

    return metrics, fig_stocks, fig_sectors