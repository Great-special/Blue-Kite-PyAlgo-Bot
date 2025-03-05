####################################################################################################################################################

import pandas as pd
import MetaTrader5 as mt5
import logging
from threading import Thread

# MT5 Timeframes
D1 = mt5.TIMEFRAME_D1
H1 = mt5.TIMEFRAME_H1
H4 = mt5.TIMEFRAME_H4
M30 = mt5.TIMEFRAME_M30
M15 = mt5.TIMEFRAME_M15
M5 = mt5.TIMEFRAME_M5

# Order Types
BUY_LIMIT = mt5.ORDER_TYPE_BUY_LIMIT
SELL_LIMIT = mt5.ORDER_TYPE_SELL_LIMIT

# Initialize MetaTrader 5
if not mt5.initialize():
    raise RuntimeError("Failed to initialize MetaTrader 5")

class MarketStructureModel:
    def __init__(self, symbol, duration=880, risk_percentage=1.0, window_size=10, reward_ratio=2.2):
        self.symbol = symbol
        self.risk_percentage = risk_percentage
        self.window_size = window_size
        self.reward_ratio = reward_ratio
        self.data = self.get_symbol_data(symbol, duration, H1)
        self.prev_high = None
        self.prev_low = None
        self.prev_swing_high = None
        self.prev_swing_low = None

        # Setup logging
        logging.basicConfig(filename='trading_log.log', level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def calculate_atr(self, window, period=14):
        """Calculate the Average True Range (ATR)."""
        high_low = window['High'] - window['Low']
        high_close = abs(window['High'] - window['Close'].shift())
        low_close = abs(window['Low'] - window['Close'].shift())
        tr = high_low.combine(high_close, max).combine(low_close, max)
        return tr.rolling(period).mean().iloc[-1]

    def identify_swing_points(self, window):
        current_high = window['High'].max()
        current_low = window['Low'].min()

        if self.prev_high is None or self.prev_low is None:
            self.prev_high, self.prev_low = current_high, current_low
            self.prev_swing_high, self.prev_swing_low = current_high, current_low
            return None, None, None

        swing_type = None
        if current_high > self.prev_high:
            swing_type = 'HH-HL' if current_low > self.prev_low else 'HH-LH'
            self.prev_swing_high = current_high
        elif current_low < self.prev_low:
            swing_type = 'LL-LH' if current_high < self.prev_high else 'LL-HL'
            self.prev_swing_low = current_low

        self.prev_high, self.prev_low = current_high, current_low
        return swing_type, current_high, current_low


    def identify_order_block(self, window):
        if len(window) < 4:
            return None

        bearish_candle = window.iloc[-4]
        rejection_candle = window.iloc[-3]
        impulsive_candle = window.iloc[-2]
        breaking_candle = window.iloc[-1]

        if self.prev_swing_high is None or self.prev_swing_low is None:
            return None

        if (bearish_candle['Close'] < bearish_candle['Open'] and
            rejection_candle['Close'] > rejection_candle['Open'] or 
            rejection_candle['Close'] < rejection_candle['Open'] and
            rejection_candle['Low'] <= bearish_candle['Low'] and
            impulsive_candle['Close'] > impulsive_candle['Open'] and
            breaking_candle['Close'] > breaking_candle['Open'] and
            breaking_candle['High'] > self.prev_swing_high):  # Ensure the breaking candle breaks the swing high
            
            impulsive_body = impulsive_candle['Close'] - impulsive_candle['Open']
            if breaking_candle['Low'] > rejection_candle['High'] + 0.5 * impulsive_body:
                return {
                    'Order Block': 'Bullish',
                    'Order Block High': rejection_candle['High'],
                    'Order Block Low': rejection_candle['Low']
                }

        if (bearish_candle['Close'] > bearish_candle['Open'] and
            rejection_candle['Close'] < rejection_candle['Open'] or 
            rejection_candle['Close'] > rejection_candle['Open'] and
            rejection_candle['High'] >= bearish_candle['High'] and
            impulsive_candle['Close'] < impulsive_candle['Open'] and
            breaking_candle['Close'] < breaking_candle['Open'] and
            breaking_candle['Low'] < self.prev_swing_low):  # Ensure the breaking candle breaks the swing low
            
            impulsive_body = impulsive_candle['Open'] - impulsive_candle['Close']
            if breaking_candle['High'] < rejection_candle['Low'] - 0.5 * impulsive_body:
                return {
                    'Order Block': 'Bearish',
                    'Order Block High': rejection_candle['High'],
                    'Order Block Low': rejection_candle['Low']
                }

        return None

    def calculate_lot_size(self, stop_loss_pips):
        """Calculate lot size based on account balance and risk percentage."""
        account_info = mt5.account_info()
        if account_info is None:
            logging.error("Failed to get account info")
            return 0

        balance = account_info.balance
        risk_amount = (self.risk_percentage / 100) * balance
        lot_size = risk_amount / (stop_loss_pips * 10)  # Assuming a standard pip value
        return lot_size

    def get_symbol_data(self, symbol, duration, frame=M15):
        """Get historical price data for the symbol."""
        rates = mt5.copy_rates_from_pos(symbol, frame, 1, duration)
        if rates is None or len(rates) == 0:
            logging.error(f"Failed to get rates for {symbol}")
            return pd.DataFrame()

        rates_frame = pd.DataFrame(rates, columns=['time', 'Open', 'High', 'Low', 'Close'])
        rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
        return rates_frame

    def place_trade(self, order_block, atr):
        """Place a trade based on the identified order block."""
        if order_block['Order Block'] == 'Bullish':
            entry_price = order_block['Order Block High']
            stop_loss = order_block['Order Block Low'] - atr
            take_profit = entry_price + abs(entry_price - stop_loss) * self.reward_ratio
            order_type = BUY_LIMIT
        else:
            entry_price = order_block['Order Block Low']
            stop_loss = order_block['Order Block High'] + atr
            take_profit = entry_price - abs(entry_price - stop_loss) * self.reward_ratio
            order_type = SELL_LIMIT

        stop_loss_pips = abs(entry_price - stop_loss) / 0.0001  # Assuming a 4-decimal place symbol
        lot_size = self.calculate_lot_size(stop_loss_pips)

        if lot_size <= 0:
            logging.error("Lot size is too small or calculation failed.")
            return

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot_size,
            "type": order_type,
            "price": entry_price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 10,
            "magic": 234000,
            "comment": "Blue Kite Order Block Strategy",
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Order failed for {self.symbol}, retcode={result.retcode}")
        else:
            logging.info(f"Order placed: {order_type} at {entry_price}, SL={stop_loss}, TP={take_profit}")

    def analyze_market(self):
        latest_order_block = None
        
        for i in range(len(self.data) - self.window_size + 1):
            window = self.data.iloc[i:i + self.window_size]
            swing_type, current_high, current_low = self.identify_swing_points(window)
            order_block = self.identify_order_block(window)
            
            if order_block:
                atr = self.calculate_atr(window)
                latest_order_block = (order_block, atr)
                logging.info(f"Identified Order Block: {order_block}")

        if latest_order_block:
            order_block, atr = latest_order_block
            self.place_trade(order_block, atr)

    def monitor_symbols(self, symbols):
        """Monitor multiple symbols concurrently."""
        threads = [Thread(target=self.run_for_symbol, args=(symbol,)) for symbol in symbols]
        for thread in threads:
            thread.start()
            logging.info(f"Thread Started: {thread}")

        for thread in threads:
            thread.join()

    def run_for_symbol(self, symbol):
        """Run market analysis for a specific symbol."""
        model = MarketStructureModel(symbol)
        model.analyze_market()
        logging.info(f"Analying Market for {symbol}")


# Example usage
def start_bot():
    symbols = ['XAUUSD', 'EURUSD',  'GBPJPY', 'NZDCHF', 'APPLE', 'GOOGLE', 'BTCUSD']
    model = MarketStructureModel('EURUSD')
    model.monitor_symbols(symbols)

def endBot():
    positions = mt5.positions_get()
    if positions == None:
        mt5.shutdown()
        quit()




import schedule
import time

def startTask():
    print('Starting....')
    # schedule.every().minutes.do(start_bot)
    schedule.every().hour.do(start_bot)
    #schedule.every(6).hours.do(Mail)
    schedule.every().friday.at("23:59").do(endBot)
    
    schedule.every().sunday.at("23:59").do(start_bot)
    
    start = True
    while start:
        schedule.run_pending()
        time.sleep(1)


if __name__ == '__main__':
    startTask()
    # start_bot()