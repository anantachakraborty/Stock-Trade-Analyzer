#!/usr/bin/env python
# coding: utf-8

# Dependencies

# In[185]:


import pandas as pd
import numpy as np
from scipy.stats import norm


# Data Loading

# In[186]:


# Load the trade log data from the CSV file
trade_data = pd.read_csv('E:/Stock Trade Analyzer/data/trade_log.csv')


# Data Exploration

# In[187]:


trade_data.head()


# In[188]:


trade_data.tail()


# In[189]:


trade_data.info()


# In[190]:


trade_data.describe()


# In[191]:


trade_data.dtypes


# In[192]:


trade_data.shape


# In[193]:


# Checking null values
trade_data.isnull().sum().max()


#  Parameter Calculation

# In[194]:


# Define initial portfolio value and risk-free interest rate
initial_portfolio_value = 6500
risk_free_rate = 0.05


# In[195]:


# Calculate trade results and add them to the DataFrame
trade_data['Trade_Result'] = trade_data['Exit Price'] - trade_data['Entry Price']


# In[196]:


# Calculate the parameters
total_trades = len(trade_data)
profitable_trades = len(trade_data[trade_data['Trade_Result'] > 0])
loss_making_trades = len(trade_data[trade_data['Trade_Result'] < 0])
win_rate = profitable_trades / total_trades

profitable_trades_data = trade_data[trade_data['Trade_Result'] > 0]
loss_making_trades_data = trade_data[trade_data['Trade_Result'] < 0]

average_profit_per_trade = profitable_trades_data['Trade_Result'].mean()
average_loss_per_trade = loss_making_trades_data['Trade_Result'].mean()
risk_reward_ratio = abs(average_profit_per_trade) / abs(average_loss_per_trade)
loss_rate = 1 - win_rate
expectancy = (win_rate * average_profit_per_trade) - (loss_rate * average_loss_per_trade)


# In[197]:


# Calculate daily returns
trade_data['Daily_Return'] = trade_data['Trade_Result'] / initial_portfolio_value


# In[198]:


# Calculate CAGR (Compound Annual Growth Rate)
ending_portfolio_value = initial_portfolio_value + trade_data['Trade_Result'].sum()
number_of_trading_days = len(trade_data)
# Assuming 252 trading days in a year
cagr = (ending_portfolio_value / initial_portfolio_value) ** (252 / number_of_trading_days) - 1  


# In[199]:


# Calculate daily returns and standard deviation
daily_rf_rate = (1 + risk_free_rate) ** (1 / 252) - 1
daily_volatility = trade_data['Daily_Return'].std()


# In[200]:


# Calculate Sharpe Ratio
sharpe_ratio = (cagr - daily_rf_rate) / daily_volatility


# In[201]:


# Calculate Max Drawdown
cumulative_returns = trade_data['Daily_Return'].cumsum()
cumulative_max = cumulative_returns.cummax()
drawdown = cumulative_max - cumulative_returns
max_drawdown = drawdown.max()
max_drawdown_percentage = (max_drawdown / cumulative_max.max()) * 100


# In[202]:


# Calculate Calmar Ratio
calmar_ratio = cagr / max_drawdown


# In[203]:


# Create a DataFrame for the results
results = pd.DataFrame({
    'Total Trades': [total_trades],
    'Profitable Trades': [profitable_trades],
    'Loss-Making Trades': [loss_making_trades],
    'Win Rate': [win_rate],
    'Average Profit per Trade': [average_profit_per_trade],
    'Average Loss per Trade': [average_loss_per_trade],
    'Risk Reward Ratio': [risk_reward_ratio],
    'Expectancy': [expectancy],
    'Average ROR per Trade': [cagr],
    'Sharpe Ratio': [sharpe_ratio],
    'Max Drawdown': [max_drawdown],
    'Max Drawdown Percentage': [max_drawdown_percentage],
    'CAGR': [cagr],
    'Calmar Ratio': [calmar_ratio]
})


# In[204]:


# Print the calculated results
for key, value in results.items():
    print(f"{key}: {value}")


#  Converting Results in CSV Format file

# In[205]:


# Create a DataFrame for the results
results = pd.DataFrame(results, index=[0])

# Save the results to a CSV file
results.to_csv('trade_results.csv', index=False)

