'''
Author: hibana2077 hibana2077@gmail.com
Date: 2023-06-25 20:38:59
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2023-06-28 14:09:55
FilePath: \fintech_studies\ai_tpsl\get_csv.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import ccxt
import re
import logging
import pandas as pd
import datetime as dt

#logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


mexc = ccxt.binanceusdm({
    'options': {
        'defaultType': 'swap',
    },
})

symbol = input('input symbol(eg: BTC/USDT:USDT): ')
timeframe = input('input timeframe(eg: 5m): ')
start_time = input('input start time(eg: 2023-06-01 00:00:00): ')
end_time = input('input end time(eg: 2023-06-05 00:00:00): ')

log_format = "\033[1;36m" + "{:=^50}" + "\033[0m"  # Cyan color
log_content_format = "\033[1;33m" + "{:^10}" + "\033[0m"  # Yellow color
log_success_format = "\033[1;32m" + "{:^10}" + "\033[0m"  # Green color
log_loding_format = "\033[1;34m" + "{:^10}" + "\033[0m"  # Blue color

logging.info(log_format.format(' Input Info '))
logging.info('Symbol: ' + log_content_format.format(symbol))
logging.info('Timeframe: ' + log_content_format.format(timeframe))
logging.info('Start Time: ' + log_content_format.format(start_time))
logging.info('End Time: ' + log_content_format.format(end_time))
logging.info(log_format.format(' End of Input Info '))

logging.info(log_loding_format.format(' Load markets '))
mexc.load_markets()
logging.info(log_success_format.format(' Load markets success '))

logging.info(log_loding_format.format(' Check params '))
if not mexc.has['fetchOHLCV']:
    logging.error('mexc has not fetchOHLCV')
    exit(1)
if timeframe not in mexc.timeframes:
    logging.error('timeframe not in mexc.timeframes')
    exit(1)
if symbol not in mexc.symbols:
    logging.error('symbol not in mexc.symbols')
    exit(1)
logging.info(log_success_format.format(' Check params success '))

logging.info(log_loding_format.format(' Get ohlcv '))
timer_st = dt.datetime.now()
starttime = dt.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
endtime = dt.datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
temp_data = []
while True:
    ohlcv = mexc.fetch_ohlcv(symbol, timeframe, limit=2, since=int(starttime.timestamp() * 1000))
    if len(ohlcv) == 0:
        break
    temp_data += ohlcv
    starttime = dt.datetime.fromtimestamp(ohlcv[-1][0] / 1000)
    if starttime > endtime:
        break
logging.info(log_success_format.format(' Get ohlcv success , cost time: ' + str(dt.datetime.now() - timer_st)))

logging.info(log_loding_format.format(' Convert to dataframe '))
df = pd.DataFrame(temp_data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
df['date'] = pd.to_datetime(df['date'], unit='ms')
df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
df.set_index('date', inplace=True)
logging.info(log_success_format.format(' Convert to dataframe success '))

logging.info(log_loding_format.format(' Show head '))
print()
print(df.head())
print()

filename = "".join(re.sub(r'[/:]', '_', symbol).split('_')[:-1]) + '_' + timeframe + '.csv'
logging.info(log_loding_format.format(' Save to csv '))
df.to_csv(filename) 
logging.info(log_success_format.format(' Save to csv success , file name: ' + filename))