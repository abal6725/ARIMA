import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader as pdr

stock = pdr.get_data_yahoo('TSLA')
stock['Average']=(stock['High']+stock['Low'])/2
stock['Diff'] = stock['Average'].diff()

### Get weekly stock data
open = stock.Open.resample('W-MON', how='last')
close = stock.Close.resample('W-FRI', how='last').resample('W-MON', how='last')
high = stock.High.resample('W-MON', how='max')
low = stock.Low.resample('W-MON', how='min')
vol = stock.Volume.resample('W-MON', how='sum')
weekly_data = pd.concat([open, close, high, low, vol], axis=1)
weekly_data = weekly_data.drop(weekly_data.index[len(weekly_data)-1])

###Plot daily difference in stock price
stock.loc['2018-08-01':'2018-08-15', 'Diff'].plot.bar(color = 'r', label='series')

##adding weekend values
weekends = pd.Series([stock.loc['2018-08-03','Average'],stock.loc['2018-08-03','Average'],stock.loc['2018-08-10','Average'],stock.loc['2018-08-10','Average']],
                     index = pd.to_datetime(['2018-08-04','2018-08-05','2018-08-11','2018-08-12']))
Stock_Average = stock['Average'].append(weekends).sort_index()
