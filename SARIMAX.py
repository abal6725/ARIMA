

import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime
import requests
from io import BytesIO


# Dataset


# Variables
endog = weekly_data['High']
exog = sm.add_constant(np.random.choice([0.3,0.4,0.5,0.6,0.7],weekly_data.shape[0]))

## decompose

from statsmodels.tsa.seasonal import seasonal_decompose

results = seasonal_decompose(endog, model= 'multiplicative')

import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools

Observed =  go.Scatter(x=results.seasonal.index, y =results.observed)
Trend =  go.Scatter(x=results.seasonal.index, y =results.trend)
RESID =  go.Scatter(x=results.seasonal.index, y =results.resid)
Seasonal =  go.Scatter(x=results.seasonal.index, y =results.seasonal)

fig = tools.make_subplots(rows=2,cols=2,subplot_titles=['Observed','Trend','resid','Seasonal'])
fig.append_trace(Observed,1,1)
fig.append_trace(Trend,1,2)
fig.append_trace(RESID,2,1)
fig.append_trace(Seasonal,2,2)

fig['layout'].update(height= 750, width = 1000,showlegend=False, title='Tesla Stock Price Decomposition')

py.iplot(fig,filename='Tesla Stock Price Decomposition')
### get the order and seasonal order using auto_arima for the endogenous values
from pyramid.arima import auto_arima

stepwise_model = auto_arima(endog, start_p=1, start_q=1,
                           max_p=1, max_q=1, m=52,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)
print(stepwise_model.aic())

order = stepwise_model.order
seasonal_order = stepwise_model.seasonal_order

# Fit the model excluding last values, for out of sample predictions
### with exogenous values
mod = sm.tsa.statespace.SARIMAX(endog[:280], exog[:280], order=order, seasonal_order = seasonal_order)
### without exogenous values
mod = sm.tsa.statespace.SARIMAX(endog[:275], order=order, seasonal_order = seasonal_order)
fit_res = mod.fit(disp=False)
print(fit_res.summary())

### with exogenous values
predict = fit_res.get_prediction(endog = endog[276:], exog = exog[281:], start=281, end=422)
### without exogenous values
predict = fit_res.get_prediction(endog = endog[276:], start=276, end=426)
predict_ci = predict.conf_int()

# Graph
fig, ax = plt.subplots(figsize=(9,4))
npre = 4
ax.set(title='Actual vs Predicted Stock Price', xlabel='Date', ylabel='Stock Prive')

# Plot data points
weekly_data.loc['2010-07-05':, 'High'].plot(ax=ax, style='o', label='Observed')

# Plot predictions
predict.predicted_mean.loc['2013-01-07':].plot(ax=ax, style='r--', label='One-step-ahead forecast')
ci = predict_ci.loc['2013-01-07':]
ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='r', alpha=0.1)

legend = ax.legend(loc='lower right')
