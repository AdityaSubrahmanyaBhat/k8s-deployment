import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn import metrics

class TSF:
    def __init__(self) -> None:
        df=pd.read_csv('Covid\data.csv',parse_dates=True,index_col="Date")
        df=df.dropna()
        self.medium_train=df.iloc[:-20]
        self.medium_test=df.iloc[-20:]
        self.small_train=df.iloc[141:240]
        self.small_test=df.iloc[240:245]

    def train(self)->None:
        self.sarimax_medium=SARIMAX(self.medium_train["Cases"])
        self.sarimax_medium.fit()

        self.sarimax_small=SARIMAX(self.small_train["Cases"])
        self.sarimax_small.fit()

        # print(self.medium_train["Cases"])
        # self.varmax_medium=VARMAX(self.medium_train["Cases"])
        # self.varmax_medium.fit()

        # self.varmax_small=VARMAX(self.small_train["Cases"])
        # self.varmax_small.fit()
        

    def predict_(self)->None:
        start_med=len(self.medium_train)
        end_med=len(self.medium_train)+len(self.medium_test)-1
        self.sarimax_med_pred=self.sarimax_medium.predict(start=start_med,end=end_med,typ='levels')
        # self.varmax_med_pred=self.varmax_medium.predict(start=start_med,end=end_med,typ='levels')

        start_sm=len(self.small_train)
        end_sm=len(self.small_train)+len(self.small_test)-1
        self.sarimax_sm_pred=self.sarimax_small.predict(start=start_sm,end=end_sm,typ='levels')
        # self.varmax_sm_pred=self.varmax_small.predict(start=start_sm,end=end_sm,typ='levels')


    def metrics(self)->None:
        self.MAE_sarimax_med = metrics.mean_absolute_error(self.medium_test,self.sarimax_med_pred)
        self.MSE_sarimax_med = metrics.mean_squared_error(self.medium_test, self.sarimax_med_pred)
        self.RMSE_sarimax_med = np.sqrt(self.MSE_sarimax_med)

        # self.MAE_varmax_med = metrics.mean_absolute_error(self.medium_test,self.varmax_med_pred)
        # self.MSE_varmax_med = metrics.mean_squared_error(self.medium_test, self.varmax_med_pred)
        # self.RMSE_varmax_med = np.sqrt(self.MSE_varmax_med)


        self.MAE_sarimax_sm = metrics.mean_absolute_error(self.small_test,self.sarimax_sm_pred)
        self.MSE_sarimax_sm = metrics.mean_squared_error(self.small_test, self.sarimax_sm_pred)
        self.RMSE_sarimax_sm = np.sqrt(self.MSE_sarimax_sm)

        # self.MAE_varmax_sm = metrics.mean_absolute_error(self.small_test,self.varmax_sm_pred)
        # self.MSE_varmax_sm = metrics.mean_squared_error(self.small_test, self.varmax_sm_pred)
        # self.RMSE_varmax_sm = np.sqrt(self.MSE_varmax_sm)

        print("Sarimax medium")
        print(self.MAE_sarimax_med)
        print(self.RMSE_sarimax_med)
        # print("Varmax medium")
        # print(self.MAE_varmax_med)
        # print(self.RMSE_varmax_med)
        print("Sarimax small")
        print(self.MAE_sarimax_sm)
        print(self.RMSE_sarimax_sm)
        # print("Varmax small")
        # print(self.MAE_varmax_sm)
        # print(self.RMSE_varmax_sm)
        

if __name__=="__main__":
    tsf=TSF()
    tsf.train()
    tsf.predict_()
    tsf.metrics()