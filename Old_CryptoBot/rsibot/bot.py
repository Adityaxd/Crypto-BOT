import websocket, json, pprint, talib, numpy
import config
from binance.client import Client
from binance.enums import *

SOCKET = "wss://stream.binance.com:9443/ws/ethusdt@kline_1m" #Listening to Binance API for ETHUSDT values from the candlechart(Kline) at an interval of 1 mins.

# Setting for our bot to execute trades / take decisions on.

# Relative Strenght Index (RSI) indicator settings.

RSI_PERIOD = 14 
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

#Crypto Currency settings.

TRADE_SYMBOL = 'ETHUSD' #Which crypto to trade.
TRADE_QUANTITY = 0.05  #amount of quantity of crypto to trade.

closes = [] #tracking closing prices and storing in array.
in_position = False #if already in position or not.

client = Client(config.API_KEY, config.API_SECRET, tld='us') #giving our Secret Information from Binanace API to this bot for using it.

def order(side, quantity, symbol,order_type=ORDER_TYPE_MARKET):
    try:
        print("sending order")
        order = client.create_order(symbol=symbol, side=side, type=order_type, quantity=quantity)
        print(order)
    except Exception as e:
        print("an exception occured - {}".format(e))
        return False

    return True

    
def on_open(ws):
    print('opened connection')

def on_close(ws):
    print('closed connection')

def on_message(ws, message):
    global closes, in_position
    
    print('received message')
    json_message = json.loads(message) #convert out json data from binance to python data structure that we can use.
    pprint.pprint(json_message)  #just pretty print data.

    candle = json_message['k']

    is_candle_closed = candle['x']
    close = candle['c']  #checking for the closing of candle as RSI only makes use of candle close data.

    if is_candle_closed:
        print("candle closed at {}".format(close))
        closes.append(float(close))  #adding close prices to our numpy array of closing prices.
        print("closes")
        print(closes)

        if len(closes) > RSI_PERIOD:
            np_closes = numpy.array(closes) #converting our closes array to a numpy array so that TA-LIB can work on it.
            rsi = talib.RSI(np_closes, RSI_PERIOD)
            print("all rsis calculated so far")
            print(rsi)
            last_rsi = rsi[-1] #getting our last RSI value to base our calculations upon.
            print("the current rsi is {}".format(last_rsi))

            if last_rsi > RSI_OVERBOUGHT:
                if in_position:
                    print("Overbought! Sell! Sell! Sell!")
                    # put binance sell logic here
                    order_succeeded = order(SIDE_SELL, TRADE_QUANTITY, TRADE_SYMBOL)
                    if order_succeeded:
                        in_position = False
                else:
                    print("It is overbought, but we don't own any. Nothing to do.")
            
            if last_rsi < RSI_OVERSOLD:
                if in_position:
                    print("It is oversold, but you already own it, nothing to do.")
                else:
                    print("Oversold! Buy! Buy! Buy!")
                    # put binance buy order logic here
                    order_succeeded = order(SIDE_BUY, TRADE_QUANTITY, TRADE_SYMBOL)
                    if order_succeeded:
                        in_position = True

                
ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_close=on_close, on_message=on_message)
ws.run_forever()