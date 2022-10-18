from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from .const import *

class LR:
    def __init__(self) -> None:    
        df = pd.read_csv(DATASET)
        X = df.drop('Y house price of unit area', axis=1)
        y = df['Y house price of unit area']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=101
            )
    def train(self)->None:
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
    
    def predict(self)->None:
        self.y_pred = self.model.predict(self.X_test)
    
    def metrics(self)->None:
        self.MAE = metrics.mean_absolute_error(self.y_test, self.y_pred)
        self.MSE = metrics.mean_squared_error(self.y_test, self.y_pred)
        self.RMSE = np.sqrt(self.MSE)

def main()->Tuple[float, float, float]:
    lr = LR()
    lr.train()
    lr.predict()
    lr.metrics()
    return (lr.MAE, lr.MSE, lr.RMSE)

if __name__ == '__main__':
    print(main())