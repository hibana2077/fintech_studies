'''
Author: hibana2077 hibana2077@gmail.com
Date: 2023-12-21 17:34:57
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2023-12-22 00:59:20
FilePath: \fintech_studies\mamba_price_predict\test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pandas_ta as ta
import pandas as pd
import numpy as np

file_path = "./data/ETHUSDT_5m.csv"

raw_data = pd.read_csv(file_path)
raw_data.set_index('time', inplace=True)
raw_data.sort_index(inplace=True)
raw_data = raw_data.astype(float)

print(help(ta.fibonacci))