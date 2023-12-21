'''
Author: hibana2077 hibana2077@gmail.com
Date: 2023-12-21 17:25:17
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2023-12-21 17:45:58
FilePath: \fintech_studies\mamba_price_predict\data_process.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pandas as pd
import pandas_ta as ta
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('data_process')

# Set up argument parser
# input : load_data_location, save_data_location
parser = argparse.ArgumentParser(description='data process')
parser.add_argument('--load_data_location', type=str, default='data/raw_data.csv')
parser.add_argument('--save_data_location', type=str, default='data/processed_data.csv')

args = parser.parse_args()

# Load data
raw_data = pd.read_csv(args.load_data_location)
raw_data.set_index('time', inplace=True)
raw_data.sort_index(inplace=True)
raw_data = raw_data[['open', 'high', 'low', 'close', 'volume']]
raw_data = raw_data.astype(float)

# Add indicators

# Moving average
raw_data['ema_5'] = ta.ema(raw_data['close'], length=5)
raw_data['ema_10'] = ta.ema(raw_data['close'], length=10)
raw_data['ema_20'] = ta.ema(raw_data['close'], length=20)
raw_data['ema_50'] = ta.ema(raw_data['close'], length=50)
raw_data['ema_100'] = ta.ema(raw_data['close'], length=100)
raw_data['ema_200'] = ta.ema(raw_data['close'], length=200)

# MACD
temp = ta.macd(raw_data['close'])
raw_data['macd'] = temp['MACD_12_26_9']
raw_data['macd_signal'] = temp['MACDs_12_26_9']
raw_data['macd_hist'] = temp['MACDh_12_26_9']

# RSI
raw_data['rsi'] = ta.rsi(raw_data['close'], length=14)

# Stochastic
temp = ta.stoch(raw_data['high'], raw_data['low'], raw_data['close'])
raw_data['stoch_k'] = temp['STOCHk_14_3_3']
raw_data['stoch_d'] = temp['STOCHd_14_3_3']

# Entropy
raw_data['entropy'] = ta.entropy(raw_data['close'], length=10)

# Drop NaN
raw_data.dropna(inplace=True)

# Save data
raw_data.to_csv(args.save_data_location)