import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn import metrics
from matplotlib import pyplot as plt

class TSF:
    def __init__(self) -> None:
        df=pd.read_csv('TSF/Prophet/temperature.csv',parse_dates=True)
        df=df.dropna()
        self.train_data=df

    def train(self)->None:
        self.model=Prophet()
        self.train_data=self.train_data.drop(["MaxTemp","MinTemp","Sunrise","Sunset"],axis="columns")
        self.train_data.rename(columns = {'DATE':'ds'}, inplace = True)
        self.train_data.rename(columns = {'AvgTemp':'y'}, inplace = True)
        # print(self.train_data)
        self.model=self.model.fit(self.train_data)

    def predict_(self)->None:
        self.future = list()
        for i in range(1, 13):
            for j in range(1,28):
                date = '{}-{}-2018'.format(i,j)
                self.future.append([date])
        self.future = pd.DataFrame(self.future)
        self.future.columns = ['ds']
        self.future['ds']= pd.to_datetime(self.future['ds'])
        self.pred=self.model.predict(self.future)

    def metrics(self)->None:
        self.y_true = self.train_data['y'][-324:].values
        print(self.train_data['y'][-12:])
        self.y_pred = self.pred['yhat'].values
        self.MAE = metrics.mean_absolute_error(self.y_true,self.y_pred)
        self.MSE = metrics.mean_squared_error(self.y_true, self.y_pred)
        self.RMSE = np.sqrt(self.MSE)
        print("MAE = "+str(self.MAE))
        print("MSE = "+str(self.MSE))
        print("RMSE = "+str(self.RMSE))
        

if __name__=="__main__":
    tsf=TSF()
    tsf.train()
    tsf.predict_()
    tsf.metrics()