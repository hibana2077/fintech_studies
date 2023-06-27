import ccxt
import pandas as pd

binance = ccxt.mexc3({
    'options': {
        'defaultType': 'future',
    },
})

binance.load_markets()

#get 15m ohlcv
ohlcv = binance.fetch_ohlcv('TRX/USDT:USDT', timeframe='5m', limit=100)

#convert to dataframe
df = pd.DataFrame(ohlcv, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
df['date'] = pd.to_datetime(df['date'], unit='ms')

#first date and last date
print(f"First date: {df['date'].iloc[0]}")
print(f"Last date: {df['date'].iloc[-1]}")

#convert date to serial number
df['date'] = range(1, len(df) + 1)

#preview data
print(df.tail())

#save to csv
df.to_csv('btc_usdt_15m.csv', index=False)