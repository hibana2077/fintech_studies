import pandas as pd
import numpy as np
import datetime as dt
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# 讀取股票數據
data = pd.read_csv('price_data.csv')  # 替換為你的數據檔案路徑

# 準備線性回歸模型
model = LinearRegression()

# 使用日期作為特徵，股票價格作為目標變量來訓練模型
model.fit(data['date'].values.reshape(-1,1), data['close'])

# 預測價格
predicted_price = model.predict(data['date'].values.reshape(-1,1))

#show prediction
print("{:=^40}".format("Prediction"))
print(f"trendline length: {len(predicted_price)}")
print(f"Date length: {len(data['date'])}")
print(f"Close length: {len(data['close'])}")
print("{:=^40}".format("Prediction"))

#data tail
print("{:=^40}".format("Data tail"))
print(data.tail())
print("{:=^40}".format("Data tail"))

take_profit_price = predicted_price[-1] + 0.0004

# plot candlestick and trendline
fig = go.Figure(data=[go.Candlestick(x=data['date'],
                open=data['open'], high=data['high'],
                low=data['low'], close=data['close'],
                name='market data'),
                go.Scatter(x=data['date'], y=predicted_price, name='trendline', line=dict(color='blue')),
                go.Scatter(x=data['date'], y=[take_profit_price]*len(data['date']), 
                           name='take profit', line=dict(color='green', dash='dash'))
                ])

fig.update_layout(xaxis_rangeslider_visible=False)
fig.show()
