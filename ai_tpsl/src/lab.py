'''
多倉強平價格=（維持保證金-倉位保證金+開倉均價x 數量x面值）/（數量x面值）
空倉強平價格=（開倉均價x數量x面值-維持保證金+倉位保證金）/（數量x面值）
U本位維持保證金=開倉均價*張數*面值*維持保證金率 
U本位倉位保證金=開倉均價*張數*面值/杠桿倍數
維持保證金率 = 0.4%
根據以上公式，推導出python程式碼
'''
def liquidation_price(open_price, quantity, face_value, leverage)->tuple:
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
    MAINTENANCE_MARGIN_RATE = 0.004  # 維持保證金率 = 0.4%

    # Calculate U本位維持保證金 and U本位倉位保證金
    maintenance_margin = open_price * quantity * face_value * MAINTENANCE_MARGIN_RATE
    position_margin = open_price * quantity * face_value / leverage

    # Calculate 多倉強平價格 and 空倉強平價格
    long_liquidation_price = (maintenance_margin - position_margin + open_price * quantity * face_value) / (quantity * face_value)
    short_liquidation_price = (open_price * quantity * face_value - maintenance_margin + position_margin) / (quantity * face_value)

    return long_liquidation_price, short_liquidation_price

# print(liquidation_price(8000, 10000, 0.0001, 25))#(7712.0, 8288.0)
# print(liquidation_price(80, 10, 1, 25))#(77.12, 82.88)
import tensorflow as tf

if tf.config.list_physical_devices('GPU'):
    print('Running on GPU')
    print('GPU:', tf.config.list_physical_devices('GPU'))
else:
    print('Running on CPU')

