
import pandas as pd
import numpy as np
import os
import sys
import time
import datetime
import math
import argparse
import re
import logging

#logging
log_format = "\033[1;36m" + "{:=^50}" + "\033[0m"  # Cyan color
log_content_format = "\033[1;33m" + "{:^10}" + "\033[0m"  # Yellow color
log_success_format = "\033[1;32m" + "{:^10}" + "\033[0m"  # Green color
log_loding_format = "\033[1;34m" + "{:^10}" + "\033[0m"  # Blue color
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#argparse
parser = argparse.ArgumentParser(description='Make label for data')
parser.add_argument('-i', '--input', type=str, help='input file name', required=True)
parser.add_argument('-l', '--leverage', type=int, help='leverage', required=True)
parser.add_argument('-q', '--quantity', type=float, help='quantity', required=True)
parser.add_argument('-t', '--threshold', type=float, help='SL threshold (eg. 0.5% -> 0.005)', required=True)#percent

#input info
args = parser.parse_args()
input_file:str = args.input
leverage:int = args.leverage
quantity:float = args.quantity
threshold:float = args.threshold
if input_file == '':
    logging.error('input file is empty')
    exit(1)
if input_file.split('.')[-1] != 'csv':
    logging.error('input file is not csv')
    exit(1)
if not os.path.exists(input_file):
    logging.error('input file not exists')
    exit(1)
if leverage <= 0:
    logging.error('leverage must be positive')
    exit(1)
if quantity <= 0:
    logging.error('quantity must be positive')
    exit(1)
if threshold <= 0:
    logging.error('threshold must be positive')
    exit(1)
logging.info(log_format.format(' Input Info '))
logging.info('Input File: ' + log_content_format.format(input_file))
logging.info('Leverage: ' + log_content_format.format(str(leverage)))
logging.info('Quantity: ' + log_content_format.format(str(quantity)))
logging.info('Threshold: ' + log_content_format.format(str(threshold)))
logging.info(log_format.format(' End of Input Info '))

#read csv
logging.info(log_loding_format.format(' Read csv '))
df = pd.read_csv(input_file)
logging.info(log_success_format.format(' Read csv success '))

#show data info
logging.info(log_format.format(' Data Info '))
logging.info('Data Shape: ' + log_content_format.format(str(df.shape)))
logging.info('Data Head: \n' + log_content_format.format(str(df.head())))
logging.info(log_format.format(' End of Data Info '))

#make label
'''
這是要用於訓練模型可以給出更好的TP/SL的label
將每一根K都視為入場點
TP想要在不被強平的前提下盡可能的高
SL想要在不被強平的前提下且可以完成TP3的最低值
先找出強平點和時間點
然後在強平點之前找出最高點以及最高點之前的最低點
Label:
    TP1: percent of TP1 (original percent)
    TP2: percent of TP2 (original percent)
    TP3: percent of TP3 (original percent)
    SL: percent of SL (original percent)
Rule:
    TP1:
'''
def calculate_liquidation_price(open_price, quantity, face_value, leverage,mma=0.004)->tuple:
    '''
    ## param
        open_price: float (開倉均價)
        quantity: float (數量)
        face_value: float (面值)
        leverage: int (杠桿倍數)
        
    ## return
        long_liquidation_price: float (多倉強平價格)
        short_liquidation_price: float (空倉強平價格)

    ## example

        某一用戶以8000USDT的價格買入BTCUSDT永續合約10000張，起始槓桿倍數是25x，且為多倉。 （假設倉位10000張處於風險限制第一檔，維持保證金率為0.4%），一張面值約0.0001BTC，則該用戶的多倉強平價格為：

    ```python
        print(liquidation_price(8000, 10000, 0.0001, 25)) #(7712.0, 8288.0)
    ```

        某一用戶以80USDT的價格買入BCHUSDT永續合約10顆BCH，起始槓桿倍數是25x，且為多倉。 （假設倉位10顆BCH處於風險限制第一檔，維持保證金率為0.4%），則該用戶的多倉強平價格為：

    ```python
        print(liquidation_price(80, 10, 1, 25)) #(77.12, 82.88)
    ```
    '''
    MAINTENANCE_MARGIN_RATE = mma

    # Calculate U本位維持保證金 and U本位倉位保證金
    maintenance_margin = open_price * quantity * face_value * MAINTENANCE_MARGIN_RATE
    position_margin = open_price * quantity * face_value / leverage

    # Calculate 多倉強平價格 and 空倉強平價格
    long_liquidation_price = (maintenance_margin - position_margin + open_price * quantity * face_value) / (quantity * face_value)
    short_liquidation_price = (open_price * quantity * face_value - maintenance_margin + position_margin) / (quantity * face_value)

    return long_liquidation_price, short_liquidation_price

logging.info(log_loding_format.format(' Make label '))
temp_df = df.copy()
df_len = len(df)
for kline in range(df_len):
    #calculate liquidation price
    liquidation_price = calculate_liquidation_price(df['open'][kline],quantity,1,leverage)[0]
    #find liquidation point
    liquidation_point = 0
    for i in range(kline, df_len):
        if df['low'][i] <= liquidation_price:
            liquidation_point = i
            break
    liquidation_point = df_len if liquidation_point == 0 else liquidation_point
    #rec best TP and SL
    #rec_list -> (temp_high, temp_low)
    rec_list = list()
    temp_high = df.iloc[kline]['high']+1e-10
    temp_low = df.iloc[kline]['low']+1e-10
    rec_list.append((temp_high, temp_low))
    for i in range(kline, liquidation_point):
        need_update = False
        if df.iloc[i]['high'] > temp_high:temp_high,need_update = df.iloc[i]['high'],True
        if df.iloc[i]['low'] < temp_low:temp_low,need_update = df.iloc[i]['low'],True
        if need_update:rec_list.append((temp_high, temp_low))
        #print progress
        #if i % 1000 == 0:print('inner progress: ' + str(i) + '/' + str(df_len))
    #calculate best TP and SL
    #formula TP_percent = (TP_price - entry_price) / entry_price
    #formula SL_percent = (entry_price - SL_price) / entry_price
    #formula score = TP_percent / SL_percent (the bigger the better)
    #formula TP1,TP2,TP3 = TP_percent/3,TP_percent/2,TP_percent
    # print(f"idx: {kline} , rec_list: {rec_list} , liquidation_price: {liquidation_price} , df.iloc[kline]['open']: {df.iloc[kline]['open']}")
    # print(f"idx: {kline}")
    best_TP , best_SL = sorted(rec_list, key=lambda x: ((x[0] - df.iloc[kline]['open'])+1e-10) / df.iloc[kline]['open'] / ((df.iloc[kline]['open'] - x[1])+1e-10) / df.iloc[kline]['open'], reverse=True)[0]
    # print(f"best_TP: {best_TP} , best_SL: {best_SL}")
    #rec label
    temp_df.loc[kline, 'TP1'] = (best_TP - df.iloc[kline]['open']) / df.iloc[kline]['open'] / 3
    temp_df.loc[kline, 'TP2'] = (best_TP - df.iloc[kline]['open']) / df.iloc[kline]['open'] / 2
    temp_df.loc[kline, 'TP3'] = (best_TP - df.iloc[kline]['open']) / df.iloc[kline]['open']
    temp_df.loc[kline, 'SL'] = -1 * (((df.iloc[kline]['open'] - best_SL) / df.iloc[kline]['open']) + threshold)
    #print progress
    if kline % 1000 == 0:print('progress: ' + str(kline) + '/' + str(df_len))

logging.info(log_success_format.format(' Make label success '))

#show data info
logging.info(log_format.format(' Data Info '))
logging.info('Data Shape: ' + log_content_format.format(str(temp_df.shape)))
logging.info('Data Head: \n' + log_content_format.format(str(temp_df.head())))
logging.info(log_format.format(' End of Data Info '))

#save csv
output_file = input_file.split('.')[0] if not input_file.startswith('./') else input_file[2:].split('.')[0]
logging.info(log_loding_format.format(' Save csv '))
temp_df.to_csv(output_file + '_label.csv', index=False)
logging.info(log_success_format.format(' Save csv success '))