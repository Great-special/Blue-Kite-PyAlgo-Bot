import numpy as np
import pandas as pd
import ccxt as xt
import logging

import itertools
import schedule
import datetime
import time
import pmdarima as pm
import warnings
import pandas_ta as ta

from db import insert_order_block, get_used_order_blocks
from threading import Thread
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import het_arch
from arch import arch_model
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


# Automatic order selection for ARIMA
warnings.filterwarnings("ignore")


# instantiation of ccxt
exchange = xt.binance()
exchange.features

# TimeFrame
# D1 = mt5.TIMEFRAME_D1
# H1 = mt5.TIMEFRAME_H1
# H4 = mt5.TIMEFRAME_H4
# M30 = mt5.TIMEFRAME_M30
# M15 = mt5.TIMEFRAME_M15
# M5 = mt5.TIMEFRAME_M5
# # Order Type
# BUY = mt5.ORDER_TYPE_BUY
# BUY_LIMIT = mt5.ORDER_TYPE_BUY_LIMIT
# SELL = mt5.ORDER_TYPE_SELL
# SELL_LIMIT = mt5.ORDER_TYPE_SELL_LIMIT


class BlueKiteModel:
    
    def __init__(self, symbol, mode='test', time_frame=H1, duration=880, risk_percentage=1.0, window_size=10, reward_ratio=2.2):
        print('-'*20, symbol, '-'*20)
        self.mode =  mode
        self.symbol = symbol
        self.time_frame = time_frame
        self.symbol_info = mt5.symbol_info(self.symbol)
        self.lot_size = self.symbol_info.volume_min
        self.account_info = mt5.account_info() 
        self.risk_percentage = risk_percentage
        self.window_size = window_size
        self.reward_ratio = reward_ratio
        self.data = self.get_symbol_data(symbol, duration, self.time_frame)
        self.prev_high = None
        self.prev_low = None
        self.new_high = None
        self.new_low = None
        # print(self.data.head())
        self.best_arima_order = pm.auto_arima(self.data['Close'], seasonal=False, stepwise=True, trace=True).order
        self.arima_residuals = self.arima_model_residual(self.data, self.best_arima_order)
        self.best_garch_order = self.auto_garch_order(self.arima_residuals)
        self.garch_forecast = self.garch_model_volatility(self.arima_residuals, self.best_garch_order)
        self.context = self.train_models(self.data, mode=self.mode)
        self.context_data_frame = self.predict_trade(self.context)
        self.combined_trend_forecast, self.slope = self.combine_forecast(self.context_data_frame['close_forecast'], self.garch_forecast, self.data, mode=self.mode)
        self.recent_order_block = None 
        self.used_order_block = self.load_order_blocks(self.symbol)
        
        
        # Setup logging
        logging.basicConfig(filename='trading_log.log', level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')


    def load_order_blocks(self,symbol):
        blocks = get_used_order_blocks(symbol=symbol)
        print("[][]", blocks)
        if blocks != []:
            return blocks
        else:
            return None
    
    def calculate_atr(self, window, period=2):
        """Calculate the Average True Range (ATR)."""
        return window['ATR'].iloc[-1]

    def identify_swing_points(self, window):
        current_high = window['High'].max()
        current_low = window['Low'].min()

        if self.prev_high is None or self.prev_low is None:
            self.prev_high, self.prev_low = current_high, current_low
            self.new_high, self.new_low = current_high, current_low
            return None, None, None

        swing_type = None
        if current_high > self.prev_high:
            swing_type = 'HH-HL' if current_low > self.prev_low else 'HH-LH'
            self.new_high = current_high
            
        elif current_low < self.prev_low:
            swing_type = 'LL-LH' if current_high < self.prev_high else 'LL-HL'
            self.new_low = current_low

        self.prev_high, self.prev_low = current_high, current_low
        return swing_type, current_high, current_low


    def identify_order_block(self, window):
        if len(window) < 5:
            return None

        descend_bear_candle = window.iloc[-5]
        bearish_candle = window.iloc[-4]
        rejection_candle = window.iloc[-3]
        impulsive_candle = window.iloc[-2]
        breaking_candle = window.iloc[-1]

        if self.new_high is None or self.new_low is None:
            return None

        if (descend_bear_candle['Close'] < descend_bear_candle['Open'] and 
            descend_bear_candle['Close'] > bearish_candle['Close'] and
            bearish_candle['Close'] < bearish_candle['Open'] and
            rejection_candle['Close'] > rejection_candle['Open'] or 
            rejection_candle['Close'] < rejection_candle['Open'] and
            rejection_candle['Low'] <= bearish_candle['Low'] and
            impulsive_candle['Close'] > impulsive_candle['Open'] and
            impulsive_candle['Close'] > bearish_candle['Open'] and
            breaking_candle['Close'] > breaking_candle['Open'] and
            breaking_candle['Close'] > descend_bear_candle['High']):  # Ensure the breaking candle breaks the swing high
            
            impulsive_body = impulsive_candle['Close'] - impulsive_candle['Open']
            if breaking_candle['Low'] > rejection_candle['High'] + (0.5 * impulsive_body):
                return {
                    'Order Block': 'Bullish',
                    'Order Block High': rejection_candle['High'],
                    'Order Block Low': rejection_candle['Low'],
                    'Order Block Time': rejection_candle['time'],
                    'Order Block Location':rejection_candle.name, # getting the id or index of the series
                }

        if (descend_bear_candle['Close'] > descend_bear_candle['Open'] and
            descend_bear_candle['Close'] < bearish_candle['Close'] and
            bearish_candle['Close'] > bearish_candle['Open'] and
            rejection_candle['Close'] < rejection_candle['Open'] or 
            rejection_candle['Close'] > rejection_candle['Open'] and
            rejection_candle['High'] >= bearish_candle['High'] and
            impulsive_candle['Close'] < impulsive_candle['Open'] and
            impulsive_candle['Close'] < breaking_candle['Open'] and
            breaking_candle['Close'] < breaking_candle['Open'] and
            breaking_candle['Low'] < descend_bear_candle['Low']):  # Ensure the breaking candle breaks the swing low
            
            impulsive_body = impulsive_candle['Open'] - impulsive_candle['Close']
            if breaking_candle['High'] < rejection_candle['Low'] - (0.5 * impulsive_body):
                return {
                    'Order Block': 'Bearish',
                    'Order Block High': rejection_candle['High'],
                    'Order Block Low': rejection_candle['Low'],
                    'Order Block Time': rejection_candle['time'],
                    'Order Block Location':rejection_candle.name, # getting the id or index of the series
                }

        return None
    
    
    def auto_garch_order(self, residuals):
        best_garch_aic = float("inf")
        
        for p in range(1, 4):
            for q in range(1, 4):
                try:
                    garch_temp = arch_model(residuals, vol='Garch', p=p, q=q).fit(disp='off')
                    if garch_temp.aic < best_garch_aic:
                        best_garch_aic = garch_temp.aic
                        best_garch_order = (p, q)
                except:
                    continue

        print(f"Best GARCH order: {best_garch_order}")
        return best_garch_order

    
    def arima_model_residual(self, data, best_order, price='Close', steps=30):
        arima_model = ARIMA(data[price], order=best_order).fit()
        arima_residuals = arima_model.resid
        arima_forecast = arima_model.forecast(steps=steps)
        # print(arima_forecast)
        return arima_residuals
    
    def garch_model_volatility(self, residuals, best_garch_order, horizon=30):
        # Fit the best GARCH model
        garch_model = arch_model(residuals, vol='Garch', p=best_garch_order[0], q=best_garch_order[1]).fit(disp='off')
        garch_forecast = garch_model.forecast(horizon=horizon)
        return garch_forecast
    
    def get_symbol_data(self, symbol, duration, frame=M15):
        """Get historical price data for the symbol."""
        rates = mt5.copy_rates_from_pos(symbol, frame, 1, duration)
        if rates is None or len(rates) == 0:
            logging.error(f"Failed to get rates for {symbol}")
            return pd.DataFrame()
        
        rates_frame = pd.DataFrame(rates)
        rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
        rates_frame = rates_frame.rename(columns={'open':'Open', 'high':'High', 'low':'Low', 'close':'Close'})
        print(rates_frame)
        return rates_frame
    
    
    def analyze_data(self, data, period=30):
        prophet_df = data[['time', 'Open', 'High', 'Low', 'Close']]
        df_close = prophet_df[['time', 'Close']].rename(columns={'time':'ds', 'Close':'y'})
        df_open = prophet_df[['time', 'Open']].rename(columns={'time':'ds', 'Open':'y'})
        
        model_close = Prophet()
        model_close.fit(df_close)
        model_open = Prophet()
        model_open.fit(df_open)
        
        if self.time_frame == M15:
            future_close = model_close.make_future_dataframe(periods=period, freq='15min', include_history=False)
            future_open = model_open.make_future_dataframe(periods=period, freq='15min', include_history=False)
        
        if self.time_frame == H1:
            future_close = model_close.make_future_dataframe(periods=period, freq='60min', include_history=False)
            future_open = model_open.make_future_dataframe(periods=period, freq='60min', include_history=False)
        
        
        print(f"Number of rows in future_close: {len(future_close)}")
        
        context = {
            'model_close':model_close,
            'future_close':future_close,
            'model_open':model_open,
            'future_open':future_open,
        }
        return context


    def train_models(self, data, mode, period=30):
        if mode != "live":
            train_len = int(len(data)*0.7)
            train = data[:train_len]
            period = abs(len(data) - train_len)
            return self.analyze_data(train, period)
        else:
            return self.analyze_data(data, period)
    
    
    def predict_trade(self, context):
        model_close = context['model_close']
        model_open = context['model_open']
        future_close = context['future_close']
        future_open = context['future_open']

        
        close_forecast = model_close.predict(future_close)
        open_forecast = model_open.predict(future_open)

        context_data = {
            'time':open_forecast['ds'],
            'open_forecast':open_forecast['yhat'],
            'close_forecast':close_forecast['yhat'],
        }
        return pd.DataFrame(context_data)
    
    def combine_forecast(self, prophet_forecast, garch_forecast, data, price='Close', horizon=30, mode='test'):
        if not mode == "live":
            print("Shape of prophet_forecast.values:", prophet_forecast.values.shape)
            print("Shape of garch_forecast.mean:", garch_forecast.variance['h.01'].shape)
            
            # Expand garch_forecast_mean to match the shape of prophet_forecast_values
            garch_forecast_mean_expanded = np.repeat(garch_forecast.variance['h.01'], prophet_forecast.values.shape[0])

            # Verify shapes
            print("Shape of prophet_forecast.values:", prophet_forecast.values.shape)
            print("Shape of garch_forecast.mean (expanded):", garch_forecast_mean_expanded.shape)

            # Combine arrays
            X_full = np.vstack([prophet_forecast.values, garch_forecast_mean_expanded]).T
            print("Combined Forecast Shape:", X_full.shape)
            # print(X_full)
                
            # Prepare combined dataset for machine learning
            test_len = int(len(data)*0.7)
            y_full = data[price][test_len:].values

            # Train-test split
            split_index = len(X_full) - horizon
            X_train, X_test = X_full[:split_index], X_full[split_index:]
            y_train, y_test = y_full[:split_index], y_full[split_index:]


            # Train the model
            model = LinearRegression()
            model.fit(X_train, y_train)


            # Predict
            combined_forecast_ml = model.predict(X_test)
            # print(combined_forecast_ml)
            
            # Evaluation
            actual = y_test
            mae_ml = mean_absolute_error(actual, combined_forecast_ml)
            mse_ml = mean_squared_error(actual, combined_forecast_ml)
            rmse_ml = np.sqrt(mse_ml)
            mape_ml = mean_absolute_percentage_error(actual, combined_forecast_ml)
            accuracy = 100 * (1 -mape_ml)

            print(f'MAE (ML): {mae_ml}, MSE (ML): {mse_ml}, RMSE (ML): {rmse_ml}')
            print(f'Accuracy (ML): {accuracy:.2f}')
            print('Model Slope', model.coef_[0])
            
            return pd.DataFrame({'combined_forecast':combined_forecast_ml}), model.coef_[0]
        else:
            # Expand garch_forecast_mean to match the shape of prophet_forecast_values
            garch_forecast_mean_expanded = np.repeat(garch_forecast.variance['h.01'], prophet_forecast.values.shape[0])

            # Verify shapes
            print("Shape of prophet_forecast.values:", prophet_forecast.values.shape)
            print("Shape of garch_forecast.mean (expanded):", garch_forecast_mean_expanded.shape)

            # Combine arrays
            X = np.vstack([prophet_forecast.values, garch_forecast_mean_expanded]).T
            print("Combined Forecast Shape:", X.shape)
            # print(X)
            
            # Prepare combined dataset for machine learning
            
            y = data[price][abs(len(X)-len(data)):].values
            print(y.shape)
            # Train the model
            model = LinearRegression()
            model.fit(X, y)
            # Predict
            combined_forecast_ml = model.predict(X)
            # print(combined_forecast_ml)
            print('Model Slope', model.coef_[0])
            
            return pd.DataFrame({'combined_forecast':combined_forecast_ml}), model.coef_[0]


    def calculate_supertrend(self, period=7, multiplier=3):
        # Calculate HL2
        hl2 = (self.data['High'] + self.data['Low']) / 2
        
        # Initialize columns
        self.data['BasicUpperBand'] = hl2 + (multiplier * self.data['ATR'])
        self.data['BasicLowerBand'] = hl2 - (multiplier * self.data['ATR'])
        self.data['FinalUpperBand'] = self.data['BasicUpperBand']
        self.data['FinalLowerBand'] = self.data['BasicLowerBand']
        self.data['SuperTrend'] = np.nan

        # Calculate Final Bands and SuperTrend
        for i in range(1, len(self.data)):
            if self.data['Close'].iloc[i-1] > self.data['FinalUpperBand'].iloc[i-1]:
                self.data['FinalUpperBand'].iloc[i] = min(self.data['BasicUpperBand'].iloc[i], self.data['FinalUpperBand'].iloc[i-1])
            else:
                self.data['FinalUpperBand'].iloc[i] = self.data['BasicUpperBand'].iloc[i]
                
            if self.data['Close'].iloc[i-1] < self.data['FinalLowerBand'].iloc[i-1]:
                self.data['FinalLowerBand'].iloc[i] = max(self.data['BasicLowerBand'].iloc[i], self.data['FinalLowerBand'].iloc[i-1])
            else:
                self.data['FinalLowerBand'].iloc[i] = self.data['BasicLowerBand'].iloc[i]

            if self.data['Close'].iloc[i] > self.data['FinalUpperBand'].iloc[i]:
                self.data['SuperTrend'].iloc[i] = self.data['FinalLowerBand'].iloc[i]
            elif self.data['Close'].iloc[i] < self.data['FinalLowerBand'].iloc[i]:
                self.data['SuperTrend'].iloc[i] = self.data['FinalUpperBand'].iloc[i]

        return self.data['SuperTrend']



    def calculate_rbreaker(self, pivot_period=14):
        """Calculate R-breaker pivot and resistance/support levels."""
        self.data['Pivot'] = (self.data['High'].rolling(window=pivot_period).mean() +
                            self.data['Low'].rolling(window=pivot_period).mean() +
                            self.data['Close'].rolling(window=pivot_period).mean()) / 3
        self.data['R1'] = 2 * self.data['Pivot'] - self.data['Low'].rolling(window=pivot_period).mean()
        self.data['S1'] = 2 * self.data['Pivot'] - self.data['High'].rolling(window=pivot_period).mean()
        self.data['R2'] = self.data['Pivot'] + (self.data['High'].rolling(window=pivot_period).mean() - self.data['Low'].rolling(window=pivot_period).mean())
        self.data['S2'] = self.data['Pivot'] - (self.data['High'].rolling(window=pivot_period).mean() - self.data['Low'].rolling(window=pivot_period).mean())
        self.data['R3'] = self.data['High'].rolling(window=pivot_period).mean() + 2 * (self.data['Pivot'] - self.data['Low'].rolling(window=pivot_period).mean())
        self.data['S3'] = self.data['Low'].rolling(window=pivot_period).mean() - 2 * (self.data['High'].rolling(window=pivot_period).mean() - self.data['Pivot'])
        
        return self.data[['Pivot', 'R1', 'S1', 'R2', 'S2', 'R3', 'S3']]

    
    def create_request(self, symbol, lot_size, order_type, price, sl, tp, comment):
        # Retrieve symbol information
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"Symbol {symbol} not found")
            return None
        
        deviation = 5

        request = {
            "action": mt5.TRADE_ACTION_DEAL if order_type == BUY or order_type == SELL else mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": deviation,
            "magic": 115,  # Magic number should be an integer
            "comment": f"blue kite strategy {comment}",
            "type_time": mt5.ORDER_TIME_GTC,
            
        }
        return request


    def send_order(self, symbol, lot_size, order_type, entry_price, sl, tp, comment):
        # Risk Management
        account_info = self.account_info
        account_leverage = account_info.leverage
        account_balance = account_info.balance
        print('account_leverage', account_leverage)
        print('sl', sl)
        print('tp', tp)
        
        if order_type == BUY  or order_type == BUY_LIMIT:
            print("Bull", order_type)
            price = mt5.symbol_info_tick(symbol).ask if order_type == BUY else entry_price
            sl = sl
            tp = tp
            print(price)
            print(price - sl)
            amt_risk = (price - sl) * lot_size / account_leverage
            print("amt_risk", amt_risk)
            
        else:
            print("Bear", order_type)
            order_type = order_type
            price = mt5.symbol_info_tick(symbol).bid if order_type == SELL else entry_price
            sl = sl
            tp = tp
            print('p', price)
            print(sl - price)
            amt_risk = (sl - price) * lot_size / account_leverage
            print("amt_risk", amt_risk)

        risk_margin = (price * lot_size) / account_leverage
        print("risk_margin", risk_margin)
        
        if risk_margin < (account_balance * 0.02) and amt_risk < (account_balance * 0.1): # change to and later
            # Checking for open positions and number
            position_total = mt5.positions_total()
            position_symbol = mt5.positions_get(symbol=symbol)
            
            if position_total != 0 and len(position_symbol) != 0:
                print(f"Positions already open for symbol: {symbol}. Order not placed.")
                
            
            request = self.create_request(
                symbol=symbol, lot_size=lot_size, order_type=order_type, 
                price=price, sl=sl, tp=tp,
                comment=comment,
            )
            
            # res = check_order_eligibility(request)
            
            if request is None:
                print("Failed to create order request")
                
            # Send a trading request
            result = mt5.order_send(request)
            
            if result is None:
                print("Order send failed.")
                print(f"Last error: {mt5.last_error()}")
            else:
                # Check the execution result
                print(f"1. order_send(): by {symbol} {lot_size} lots at {price}")
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    print(f"2. order_send failed, retcode={result.retcode}")
                    # Request the result as a dictionary and display it element by element
                    result_dict = result._asdict()
                    for field in result_dict.keys():
                        print(f"   {field}={result_dict[field]}")
                        # If this is a trading request structure, display it element by element as well
                        if field == "request":
                            traderequest_dict = result_dict[field]._asdict()
                            for tradereq_field in traderequest_dict:
                                print(f"       traderequest: {tradereq_field}={traderequest_dict[tradereq_field]}")
                    print("shutdown() and quit")
                    mt5.shutdown()
                    quit()
                else:
                    print(f"2. order_send done, {result}")
                    print(f"   opened position with POSITION_TICKET={result.order}")
                    print(f"   sleep 2 seconds before closing position #{result.order}")
        else:
            print("Risk management criteria not met. Order not placed.")
            

    def analyze_market(self):
        # Technical indicators
        self.data['RSI'] = ta.rsi(self.data['Close'], 14)
        self.data['ATR'] = ta.atr(self.data['High'], self.data['Low'], self.data['Close'], length=14)
        self.data['20MA'] = ta.ma('sma', self.data['Close'], length=20)
        self.data['50MA'] = ta.ma('sma', self.data['Close'], length=50)
        
        
        forecasted = self.combined_trend_forecast['combined_forecast']
        first_close = forecasted.iloc[-3]
        last_close = forecasted.iloc[-1]
        
        # Check current market conditions
        current = self.data.iloc[-1]
        print(current)
        price = current['Close']
        
        # order block
        latest_order_block = None
        latest_current_high = None
        latest_current_low = None
        latest_swing_type = None
        UsedOrderBlock = False
        
        for i in range(len(self.data) - self.window_size + 1):
            window = self.data.iloc[i:i + self.window_size]
            swing_type, current_high, current_low = self.identify_swing_points(window)
            order_block = self.identify_order_block(window)
            
            if current_low:
                latest_current_low = current_low
            
            if current_high:
                latest_current_high = current_high
            
            if swing_type:
                latest_swing_type = swing_type
            
            if order_block:
                atr = self.calculate_atr(window)
                latest_order_block = (order_block, atr)
                logging.info(f"Identified Order Block: {order_block}")
        
        if latest_order_block != None:
            order_block, atr = latest_order_block
            if self.used_order_block != None:
                for block in self.used_order_block:
                    print(type(block))
                    if block[1] == self.symbol and block[2] == order_block['Order Block'] and order_block['Order Block High'] == block[3] and order_block['Order Block Low'] == block[4]:
                        UsedOrderBlock = True
            
            # Checking the order block duration
            order_duration = abs(current['time'] - order_block['Order Block Time'])
            print(order_duration, 'order_duration')
            
            # Extracting difference in terms of days, hours, and minutes
            days = order_duration.days
            hours, remainder = divmod(order_duration.seconds, 3600)
            minutes, _ = divmod(remainder, 60)
            print(days, 'days', hours, 'hours', minutes, 'minutes')            

            if minutes <= 60 and hours <= 1 and not UsedOrderBlock:
                candle_after_setup = order_block['Order Block Location'] + 2
                print(candle_after_setup, 'candle_after block')
                recent_data = self.data.iloc[candle_after_setup:]
                
                if order_block['Order Block'] == 'Bullish':
                    revisits = recent_data[(recent_data['Low'] <= order_block['Order Block High'])]
                    print(revisits)
                    if not len(revisits) > 0:
                        insert_order_block(symbol=self.symbol, 
                                order_block_type=order_block['Order Block'], 
                                high=order_block['Order Block High'], 
                                low=order_block['Order Block Low'], 
                                timestamp=order_block['Order Block Time'],
                                status="used")
                        entry_price = order_block['Order Block High']
                        stop_loss = order_block['Order Block Low'] - atr
                        take_profit = entry_price + abs(entry_price - stop_loss) * self.reward_ratio
                        order_type = BUY_LIMIT
                        
                        self.send_order(
                            symbol=self.symbol,
                            lot_size=self.lot_size,
                            order_type=order_type,
                            entry_price=entry_price,
                            sl=stop_loss,
                            tp=take_profit,
                            comment="bullish order block"
                        )
                
                else:
                    revisits = recent_data[(recent_data['High'] >= order_block['Order Block Low'])]
                    print(revisits)
                    if not len(revisits) > 0:
                        insert_order_block(symbol=self.symbol, 
                                order_block_type=order_block['Order Block'], 
                                high=order_block['Order Block High'], 
                                low=order_block['Order Block Low'], 
                                timestamp=order_block['Order Block Time'],
                                status="used")
                        entry_price = order_block['Order Block Low']
                        stop_loss = order_block['Order Block High'] + atr
                        take_profit = entry_price - abs(entry_price - stop_loss) * self.reward_ratio
                        order_type = SELL_LIMIT
                
                        self.send_order(
                            symbol=self.symbol,
                            lot_size=self.lot_size,
                            order_type=order_type,
                            entry_price=entry_price,
                            sl=stop_loss,
                            tp=take_profit,
                            comment="bearish order block",
                        )

                    self.recent_order_block = latest_order_block
              
        print(f"latest_current_high: {latest_current_high}")
        print(f"latest_current_low: {latest_current_low}")
        print(f"prev_high: {self.prev_high}")
        print(f"prev_low: {self.prev_low}")
        print(f"new_high: {self.new_high}")
        print(f"new_low: {self.new_low}")
    
        # if latest_swing_type == 'HH-HL':
        #    if self.prev_high == self.new_high:
        # rbreaker = self.calculate_rbreaker()
            
        if first_close > last_close:
            # Super Trend
            supertrend = self.calculate_supertrend()

            # Market analysis based on SuperTrend
            if supertrend.iloc[-1] < self.data['Close'].iloc[-1]: # supertrend.iloc[-1] < last_close
                entry_price = self.data['Close'].iloc[-1]
                stop_loss = supertrend.iloc[-1] - self.data['ATR'].iloc[-1]
                take_profit = entry_price + abs(entry_price - stop_loss) * self.reward_ratio
                
                self.send_order(
                    symbol=self.symbol,
                    lot_size=self.lot_size,
                    order_type=BUY,
                    entry_price=entry_price,
                    sl= stop_loss,
                    tp=take_profit,
                    comment="bullish supertrend"
                )
            
        elif first_close < last_close:
            supertrend = self.calculate_supertrend()
            if supertrend.iloc[-1] > self.data['Close'].iloc[-1]:
                entry_price = self.data['Close'].iloc[-1]
                stop_loss = supertrend.iloc[-1] + self.data['ATR'].iloc[-1]
                take_profit = entry_price - abs(entry_price - stop_loss) * self.reward_ratio
                
                self.send_order(
                    symbol=self.symbol,
                    lot_size=self.lot_size,
                    order_type=SELL,
                    entry_price=entry_price,
                    sl= stop_loss,
                    tp=take_profit,
                    comment="bearish supertrend"
                )
           
                
    def monitor_symbols(self, symbols, mode, time_frame, duration=880, risk_percentage=1.0, window_size=10, reward_ratio=2.2):
        """Monitor multiple symbols concurrently."""
        threads = [Thread(target=self.run_for_symbol, args=(symbol, mode, time_frame, duration, risk_percentage, window_size, reward_ratio)) for symbol in symbols]
        for thread in threads:
            thread.start()
            logging.info(f"Thread Started: {thread}")

        for thread in threads:
            thread.join()

    def run_for_symbol(self, symbol, mode, time_frame, duration, risk_percentage, window_size, reward_ratio):
        """Run market analysis for a specific symbol."""
        model = BlueKiteModel(symbol, mode, time_frame, duration, risk_percentage, window_size, reward_ratio)
        model.analyze_market()
        logging.info(f"Analying Market for {symbol}")


# BlueKite 11.5.0
# Define active and inactive times
ACTIVE_START_TIME = "09:30"
ACTIVE_END_TIME = "16:00"


# connect to MetaTrader 5
if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()

def start_bot():
    symbols = ['XAUUSD', 'EURUSD', 'EURGBP', 'GBPJPY', 'GBPUSD', 'APPLE', 'GOOGLE', 'BTCUSD']

    model = BlueKiteModel(symbol='EURGBP', mode='live', time_frame=M15)
    # model.analyze_market()
    model.monitor_symbols(symbols,  mode='live', time_frame=M15)



def start_job():
    # Schedule the job to run every minute during active trading hours
    schedule.every(15).minutes.do(start_bot)
    print("Trading job started")

def stop_job():
    # Cancel only the start_bot job, not the check_time_and_run job
    for job in schedule.jobs:
        if job.job_func == start_bot:
            schedule.cancel_job(job)
    print("Trading job stopped")
    
def check_time_and_run():
    current_time = datetime.now().strftime("%H:%M")
    current_day = datetime.now().strftime("%A")
    
    # Check if today is a weekday (Monday to Friday) and within active trading hours
    if current_day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        if ACTIVE_START_TIME <= current_time <= ACTIVE_END_TIME:
            for job in schedule.jobs:
                if not job.job_func == start_bot:
                    start_job()
        else:
            if schedule.jobs:
                stop_job()
    else:
        if schedule.jobs:
            stop_job()

# Schedule the check_time_and_run function to run every minute
schedule.every(1).minutes.do(check_time_and_run)

# Main loop to keep the script running
while True:
    schedule.run_pending()
    time.sleep(1)


