import pandas as pd
import matplotlib
import scipy

series = pd.read_csv('/home/abeer/Downloads/IPG2211A2N.csv', index_col=0)
series.index = pd.to_datetime(series.index)
series.columns=['Sales']
print(series.head())

import plotly
plotly.tools.set_credentials_file(username='abal6725', api_key='1EiANlzR4Vd4eYHDUI8H')

from statsmodels.tsa.seasonal import seasonal_decompose
from plotly.plotly import plot_mpl
result = seasonal_decompose(series, model='multiplicative')

from pyramid.arima import auto_arima

stepwise_model = auto_arima(series, start_p=1, start_q=1,
                           max_p=1, max_q=1, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)
print(stepwise_model.aic())

train = series.loc['1985-01-01':'2016-12-01']
test = series.loc['2017-01-01':]

stepwise_model.fit(train)
future_forecast = stepwise_model.predict(n_periods=18)
future_forecast = pd.DataFrame(future_forecast,index = test.index, columns=['Prediction'])
future_forecast = pd.concat([test,future_forecast],axis=1)

from sklearn.metrics import mean_squared_error
mean_squared_error(future_forecast['Sales'].as_matrix(), future_forecast['Prediction'].as_matrix())

