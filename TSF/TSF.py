import numpy as np
import pandas as pd
# from statsmodels.tsa.stattools import adfuller
# from pmdarima import auto_arima
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn import metrics

class TSF:
    def __init__(self) -> None:
        df=pd.read_csv('TSF/temperature.csv',parse_dates=True,index_col="DATE")
        df=df.dropna()
        self.train=df.iloc[:-30]
        self.test=df.iloc[-30:]

    def train(self)->None:
        self.model=SARIMAX(self.train["AvgTemp"],order=(1,0,5))
        self.model=self.model.fit()

    def predict(self)->None:
        start=len(self.train)
        end=len(self.train)+len(self.test)-1
        self.pred=self.model.predict(start=start,end=end,typ='levels').rename('ARIMA Predictions')

    def metrics(self)->None:
        self.MAE = metrics.mean_absolute_error(self.test, self.pred)
        self.MSE = metrics.mean_squared_error(self.test, self.pred)
        self.RMSE = np.sqrt(self.MSE)
        print("MAE = "+str(self.MAE))
        print("MSE = "+str(self.MSE))
        print("RMSE = "+str(self.RMSE))

if __name__=="__main__":
    tsf=TSF()
    tsf.train()
    tsf.predict()
    tsf.metrics()