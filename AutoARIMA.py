import pandas as pd
from pyramid.arima import auto_arima

### Find best Model params
stepwise_model = auto_arima(weekly_data['High'], start_p=1, start_q=1,
                           max_p=10, max_q=10, m=52,
                           start_P=0, seasonal=True,
                           d=1,D=1, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)
print(stepwise_model.aic())

train = weekly_data['High'].loc['1985-01-01':'2017-12-30']
test = weekly_data['High'].loc['2018-01-01':]

stepwise_model.fit(train)

future_forecast = stepwise_model.predict(n_periods=33)
future_forecast = pd.DataFrame(future_forecast,index = test.index, columns=['Prediction'])
future_forecast = pd.concat([test,future_forecast],axis=1)
print(future_forecast)
from sklearn.metrics import mean_squared_error
print(mean_squared_error(future_forecast['High'].as_matrix(), future_forecast['Prediction'].as_matrix()))

order = stepwise_model.order
seasonal_order = stepwise_model.seasonal_order
