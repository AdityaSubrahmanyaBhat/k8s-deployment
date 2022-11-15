import numpy as np
import pandas as pd
import pmdarima as pm
# from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
# import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn import metrics



# deaths_df=pd.read_csv("Covid TSF\covid.csv")
# d = deaths_df.loc[:, '1/22/20':]
# d=d.transpose()
# d=d.sum(axis=1)
# d=d.tolist()



# dataset=pd.DataFrame(columns=["ds","y"])

# dates=list(deaths_df.columns[4:])

# dates=list(pd.to_datetime(dates))
# dataset["ds"]=dates
# dataset["y"]=d
# dataset = dataset.set_index('ds')
# pd.DataFrame.to_csv(dataset,"covid_new.csv")
dataset=pd.read_csv("covid_new.csv")
dataset=dataset.diff() 
dataset = dataset.loc['2020-01-23':'2020-08-13']

start_date="2020-07-31"

train=dataset.loc[dataset.index < pd.to_datetime(start_date)]
test=dataset.loc[dataset.index >= pd.to_datetime(start_date)]


# model = pm.auto_arima(train, start_p=1, start_q=1,
#                       test='adf',       
#                       max_p=3, max_q=3,  
#                       m=1,              
#                       d=None,           
#                       seasonal=False,   
#                       start_P=0,
#                       D=0,
#                       trace=True,
#                       error_action='ignore',
#                       suppress_warnings=True,
#                       stepwise=True)
# print(model.summary())

model = SARIMAX(train, order=(2, 1, 3))

results = model.fit(disp=True)

sarimax_prediction = results.predict(start=start_date,
                                    end='2020-08-13',
                                    dynamic=False)

plt.figure(figsize=(10, 6))
l1, = plt.plot(dataset, label='Observation')
l2, = plt.plot(sarimax_prediction, label='ARIMAX')
plt.title('SARIMAX Prediction')
plt.legend(handles=[l1, l1])
# plt.show()

plt.savefig('SARIMAX Prediction', bbox_inches='tight', transparent=False)


print('SARIMAX MAE:', metrics.mean_absolute_error(sarimax_prediction, test))
# print('SARIMAX MSE:', metrics.mean_squared_error(sarimax_prediction, test))
print('SARIMAX RMSE:', np.sqrt(metrics.mean_squared_error(sarimax_prediction, test)))


#Prophet

train["ds"]=train.index.values
m=Prophet()
m.fit(train)

future = m.make_future_dataframe(periods=dataset.shape[0] - train.shape[0])
prophet_prediction = m.predict(future)
prophet_prediction = prophet_prediction.set_index('ds')
prophet_future = prophet_prediction.yhat.loc[prophet_prediction.index >= start_date]

print('Prophet MAE:', metrics.mean_absolute_error(prophet_future, test))
# print('Prophet MSE:', metrics.mean_squared_error(prophet_future, test))
print('Prophet RMSE:',np.sqrt(metrics.mean_squared_error(prophet_future, test)))