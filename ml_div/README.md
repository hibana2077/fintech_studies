# 基於LSTM的RSI背離指標判斷

## 背景

因為Tradingview上的RSI背離指標有滯後性，所以想要利用LSTM來判斷RSI背離指標，並且在背離發生時，進行買賣。

## 資料來源

1. [MEXC Global](https://www.mexc.com/)
2. [Tradingview](https://www.tradingview.com/)

## 使用

1. `pip install -r requirements.txt`
2. 打開 `train.ipynb` 並執行