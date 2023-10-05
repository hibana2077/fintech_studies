'''
Author: hibana2077 hibana2077@gmail.com
Date: 2023-10-05 11:56:08
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2023-10-05 13:13:48
FilePath: \fintech_studies\strategy_switcher\data_process.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pandas as pd
import pandas_ta as ta
import numpy as np
import argparse
import time
import logging
from scipy.stats import mode

#set up logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('data_process')

#set up argument parser

parser = argparse.ArgumentParser(description='data process')
parser.add_argument('--file_path', type=str, default='data/BTCUSDT-1d-data.csv')

args = parser.parse_args()

#set up data process

def data_process(file_path:str)->pd.DataFrame:
    pass

def calculate_indicators(df):
    df['ema_22'] = ta.ema(df['close'], length=22)
    df['upper'] = df['high'].rolling(window=22).max()
    df['lower'] = df['low'].rolling(window=22).min()
    df['support'] = df['low'].rolling(window=22).min()
    df['resistance'] = df['high'].rolling(window=22).max()
    return df

#strategy functions

