'''
Author: hibana2077 hibana2077@gmail.com
Date: 2023-12-21 17:34:57
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2023-12-21 19:46:46
FilePath: \fintech_studies\mamba_price_predict\test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pandas_ta as ta
import pandas as pd
import numpy as np

file_path = "./data/ETHUSDT_5m.csv"

raw_data = pd.read_csv(file_path)
# drop time
raw_data.drop('time', axis=1, inplace=True)
raw_data = raw_data.astype(float)

data = raw_data.copy()
windows_size = 10

def split_data(data:pd.DataFrame, window_size:int):
    X = []
    y = []
    for i in range(len(data) - window_size):
        X.append(data.iloc[i:i+window_size].values)
        y.append(data.iloc[i+window_size,0])
    return np.array(X), np.array(y)

X,y = split_data(data, windows_size)

print(f"==Result==")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"X[0]: {X[0]}")
print(f"y[0]: {y[0]}")