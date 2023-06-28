import pandas as pd
import pandas_ta as ta
import logging
import re
import os
import argparse

#logging
log_format = "\033[1;36m" + "{:=^50}" + "\033[0m"  # Cyan color
log_content_format = "\033[1;33m" + "{:^10}" + "\033[0m"  # Yellow color
log_success_format = "\033[1;32m" + "{:^10}" + "\033[0m"  # Green color
log_loding_format = "\033[1;34m" + "{:^10}" + "\033[0m"  # Blue color
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#argparse
parser = argparse.ArgumentParser(description='Add Technical Analysis to csv')
parser.add_argument('-i', '--input', type=str, help='input file name', required=True)

#input info
args = parser.parse_args()
input_file = args.input
if input_file == '':
    logging.error('input file is empty')
    exit(1)
if input_file.split('.')[-1] != 'csv':
    logging.error('input file is not csv')
    exit(1)
if not os.path.exists(input_file):
    logging.error('input file not exists')
    exit(1)
logging.info(log_format.format(' Input Info '))
logging.info('Input File: ' + log_content_format.format(input_file))
logging.info(log_format.format(' End of Input Info '))

#read csv
logging.info(log_loding_format.format(' Read csv '))
df = pd.read_csv(input_file)
logging.info(log_success_format.format(' Read csv success '))

#add ta
logging.info(log_loding_format.format(' Add Technical Analysis '))
df.ta.ema(close='close', length=20, append=True)
df.ta.ema(close='close', length=50, append=True)
df.ta.ema(close='close', length=100, append=True)
df.ta.ema(close='close', length=200, append=True)
df.ta.rsi(close='close', length=14, append=True)
df.ta.macd(close='close', append=True)
df.ta.vp(close='close', append=True)
logging.info(log_success_format.format(' Add Technical Analysis success '))

#drop low_close,mean_close,high_close,pos_volume,neg_volume,total_volume and nan
logging.info(log_loding_format.format(' Drop other columns and nan '))
df = df.drop(['low_close', 'mean_close', 'high_close', 'pos_volume', 'neg_volume', 'total_volume'], axis=1)
df = df.dropna()
logging.info(log_success_format.format(' Drop other columns and nan success '))

#data info
logging.info(log_format.format(' Data Info '))
logging.info('Data Shape: ' + log_content_format.format(str(df.shape)))
logging.info('Data Head: \n' + log_content_format.format(str(df.head())))
logging.info('Data Tail: \n' + log_content_format.format(str(df.tail())))
logging.info(log_format.format(' End of Data Info '))

#save csv
input_file = input_file.split('.')[0] if not input_file.startswith('./') else input_file[2:].split('.')[0]
logging.info(log_loding_format.format(' Save csv '))
df.to_csv(input_file+"_ta.csv", index=False)
logging.info(log_success_format.format(' Save csv success , file name: '+input_file+"_ta.csv"))