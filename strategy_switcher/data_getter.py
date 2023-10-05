'''
Author: hibana2077 hibana2077@gmail.com
Date: 2023-10-05 10:39:25
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2023-10-05 11:55:23
FilePath: \fintech_studies\strategy_switcher\data_getter.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pandas as pd
import numpy as np
import argparse
import time
import logging
from ccxt import binanceusdm,Exchange
from datetime import datetime,timedelta

#constants

LIMIT_LIST = [500,1000,1500]

#set up logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('data_getter')

#set up argument parser
# input : symbol, timeframe, start_time, end_time

parser = argparse.ArgumentParser(description='data getter')
parser.add_argument('--symbol', type=str, default='BTC/USDT')
parser.add_argument('--timeframe', type=str, default='1d')
parser.add_argument('--start_time', type=str, default='2022-01-01 00:00:00')
parser.add_argument('--end_time', type=str, default='2023-01-01 00:00:00')

args = parser.parse_args()

#set up ccxt binance exchange

exchange = binanceusdm({})
exchange.load_markets()
exchange.verbose = False

#set up time

start_time = datetime.strptime(args.start_time, '%Y-%m-%d %H:%M:%S')
end_time = datetime.strptime(args.end_time, '%Y-%m-%d %H:%M:%S')
timeframe = args.timeframe
symbol = args.symbol

#set up data getter

def timeframe_to_milliseconds(timeframe:str)->int:
    if timeframe.endswith('m'):
        return int(timeframe[:-1]) * 60 * 1000
    elif timeframe.endswith('h'):
        return int(timeframe[:-1]) * 60 * 60 * 1000
    elif timeframe.endswith('d'):
        return int(timeframe[:-1]) * 24 * 60 * 60 * 1000
    else:
        raise ValueError(f'invalid timeframe {timeframe}')

def get_data(exchange:Exchange,
             timeframe:str,
                symbol:str,
                start_time:datetime,
                end_time:datetime)->pd.DataFrame:
    
    #set up time
    start_time = start_time.timestamp() * 1000
    end_time = end_time.timestamp() * 1000

    #check best limit **has bug**
    # cnt_list = dict()
    # for limit in LIMIT_LIST:
    #     #check max kline count
    #     max_kline_count = timedelta(milliseconds=end_time - start_time) // timedelta(milliseconds=timeframe_to_milliseconds(timeframe))#return int
    #     request_times = max_kline_count // limit + 1
    #     cnt_list[limit] = request_times
    # limit = max(cnt_list, key=cnt_list.get)
    # logger.info(f'best limit is {limit}')

    #get data
    out_data = []
    logger.info(f'time info: start_time: {start_time}, end_time: {end_time}, timeframe: {timeframe}')
    while start_time < end_time:
        data = exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, since=start_time, limit=1000)
        out_data += data
        start_time = data[-1][0] + timeframe_to_milliseconds(timeframe)
        time.sleep(0.05)
    df = pd.DataFrame(out_data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df.set_index('time', inplace=True)
    return df

#set up data saver

def save_data(df:pd.DataFrame, symbol:str, timeframe:str):
    df.to_csv(f'./data/{symbol.replace("/", "")}_{timeframe}.csv')

#Basic data check

def data_checker(df:pd.DataFrame):
    #check if there is any nan value
    if df.isnull().values.any():
        logger.warning('there is nan value in the data')
    #check if ther has any duplicated index
    if df.index.duplicated().any():
        logger.warning('there is duplicated index in the data')
    #check if the index is in order
    if not df.index.is_monotonic_increasing and not df.index.is_monotonic_decreasing:
        logger.warning('the index is not in order')

#run

if __name__ == '__main__':
    logger.info('start getting data')
    df = get_data(exchange, timeframe, symbol, start_time, end_time)
    logger.info('data get')
    data_checker(df)
    logger.info('start saving data')
    save_data(df, symbol, timeframe)
    logger.info('data saved')
