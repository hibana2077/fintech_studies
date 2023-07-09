
import pandas as pd
import numpy as np
import tensorflow as tf
import os
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
def calculate(df:pd.DataFrame, quantity:float, leverage:int, threshold:float=0.01) -> pd.DataFrame:
    df_len = len(df)
    temp_df = df.copy()
    
    liquidation_prices = calculate_liquidation_price(df['open'], quantity, 1, leverage)[0]
    liquidation_points = [df_len if np.all(df['low'][i:].gt(liquidation_prices[i])) else df['low'][i:].le(liquidation_prices[i]).idxmax() for i in range(df_len)]
    
    best_TPs = []
    best_SLs = []
    for kline, lp in enumerate(liquidation_points):
        temp_high = df.iloc[kline:lp]['high'].max() if lp != df_len else df.iloc[kline]['high']
        temp_low = df.iloc[kline:lp]['low'].min() if lp != df_len else df.iloc[kline]['low']
        best_TP, best_SL = max([(temp_high, temp_low)], key=lambda x: ((x[0] - df.iloc[kline]['open'])+1e-10) / df.iloc[kline]['open'] / ((df.iloc[kline]['open'] - x[1])+1e-10) / df.iloc[kline]['open'])
        best_TPs.append(best_TP)
        best_SLs.append(best_SL)
    
    temp_df['TP1'] = (np.array(best_TPs) - df['open']) / df['open'] / 3
    temp_df['TP2'] = (np.array(best_TPs) - df['open']) / df['open'] / 2
    temp_df['TP3'] = (np.array(best_TPs) - df['open']) / df['open']
    temp_df['SL'] = -1 * (((df['open'] - np.array(best_SLs)) / df['open']) + threshold)
    
    return temp_df

temp_df = calculate(df, quantity, leverage, threshold)


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