'''
Author: hibana2077 hibana2077@gmail.com
Date: 2023-10-05 11:56:08
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2023-10-05 21:06:51
FilePath: \fintech_studies\strategy_switcher\data_process.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pandas as pd
import pandas_ta as ta
import numpy as np
import argparse
import logging

#set up logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('data_process')

#set up argument parser

parser = argparse.ArgumentParser(description='data process')
parser.add_argument('--file_path', type=str, default='data/BTCUSDT_4h.csv')

args = parser.parse_args()

#set up data process

def data_process(file_path:str)->pd.DataFrame:
    return pd.read_csv(file_path)

def calculate_indicators(df):
    df['ema_22'] = ta.ema(df['close'], length=22)
    df['ema_50'] = ta.ema(df['close'], length=50)
    df['ema_200'] = ta.ema(df['close'], length=200)
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['kurtosis'] = ta.kurtosis(df['close'], length=50)
    df['mad'] = ta.mad(df['close'], length=50)
    df['tos_stdevall'] = ta.stdev(df['close'], length=50)
    return df

#strategy functions

def cross_up_float(series1:pd.Series, threshold:float)->pd.Series:
    return (series1.shift(1) < threshold) & (series1 > threshold)

def cross_down_float(series1:pd.Series, threshold:float)->pd.Series:
    return (series1.shift(1) > threshold) & (series1 < threshold)

def cross_up_series(series1:pd.Series, series2:pd.Series)->pd.Series:
    return (series1.shift(1) < series2.shift(1)) & (series1 > series2)  

def cross_down_series(series1:pd.Series, series2:pd.Series)->pd.Series:
    return (series1.shift(1) > series2.shift(1)) & (series1 < series2)

def cross_up_int(series1:pd.Series, threshold:int)->pd.Series:
    return (series1.shift(1) < threshold) & (series1 > threshold)

def cross_down_int(series1:pd.Series, threshold:int)->pd.Series:
    return (series1.shift(1) > threshold) & (series1 < threshold)

def strategy_1(df:pd.DataFrame)->pd.DataFrame:
    '''
    if ema_22 crosses over ema_50 and kurtosis over 0, then buy
    if ema_22 crosses down ema_50 and kurtosis under 0, then sell
    '''
    df = calculate_indicators(df)
    df['signal1'] = np.where(cross_up_series(df['ema_22'], df['ema_50']) & (df['kurtosis'] > 0), 1, np.nan)
    df['signal1'] = np.where(cross_down_series(df['ema_22'], df['ema_50']) & (df['kurtosis'] < 0), -1, df['signal1'])
    df['signal1'] = df['signal1'].fillna(0)
    return df

def strategy_2(df:pd.DataFrame)->pd.DataFrame:
    '''
    if rsi crosses down 30 and kurtosis over 0, then buy
    if rsi crosses up 70 and kurtosis under 0, then sell
    '''
    df = calculate_indicators(df)
    df['signal2'] = np.where(cross_down_int(df['rsi'], 30) & (df['kurtosis'] > 0), 1, np.nan)
    df['signal2'] = np.where(cross_up_int(df['rsi'], 70) & (df['kurtosis'] < 0), -1, df['signal2'])
    df['signal2'] = df['signal2'].fillna(0)
    return df

#main

if __name__ == '__main__':
    logging.info('start data process')
    df = data_process(args.file_path)
    logging.info('start strategy 1')
    df = strategy_1(df)
    logging.info('start strategy 2')
    df = strategy_2(df)
    logging.info('start saving')
    save_name = args.file_path.split('.')[0] + '_processed.csv'
    logging.info(f'save to {save_name}')
    df.to_csv(save_name,index=False)
    logging.info('done')