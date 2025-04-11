import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import logging
import joblib
import time

import schedule
import datetime
import pmdarima as pm
import warnings
import pandas_ta as ta

from db import insert_order_block, get_used_order_blocks
from threading import Thread

# Machine Learning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import het_arch
from arch import arch_model
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, classification_report, r2_score

# Web scraping
from news_data import get_news_data
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk


nltk.download('vader_lexicon')

# Automatic order selection for ARIMA
warnings.filterwarnings("ignore")


# TimeFrame
D1 = mt5.TIMEFRAME_D1
H1 = mt5.TIMEFRAME_H1
H4 = mt5.TIMEFRAME_H4
M30 = mt5.TIMEFRAME_M30
M15 = mt5.TIMEFRAME_M15
M5 = mt5.TIMEFRAME_M5
# Order Type
BUY = mt5.ORDER_TYPE_BUY
BUY_LIMIT = mt5.ORDER_TYPE_BUY_LIMIT
SELL = mt5.ORDER_TYPE_SELL
SELL_LIMIT = mt5.ORDER_TYPE_SELL_LIMIT


class BlueKiteModel:
    
    def __init__(self, symbol, mode='test', time_frame=H1, duration=1880, risk_percentage=1.0, window_size=10, reward_ratio=1.8, retrain=True, risk_level='Medium'):
        
        if risk_level.lower() in ['high', 'low', 'medium']:
            self.risk_level = risk_level
        else:
            raise ValueError("Risk level must be 'High', 'Medium', or 'Low'.")
        #print('-'*20, symbol, '-'*20)
        self.mode =  mode
        self.symbol = symbol
        self.time_frame = time_frame
        self.symbol_info = mt5.symbol_info(self.symbol)
        self.retrain = retrain
        try:
            self.lot_size = self.symbol_info.volume_min 
        except:
            self.lot_size = 0.01
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
        if retrain:
            self.best_arima_order = pm.auto_arima(self.data['Close'], seasonal=False, stepwise=True, trace=True).order
            self.arima_residuals = self.arima_model_residual(self.data, self.best_arima_order)
            self.best_garch_order = self.auto_garch_order(self.arima_residuals)
            self.garch_model = self.garch_model_volatility(self.arima_residuals, self.best_garch_order)
            self.context = self.prophet_train_models(self.data, mode=self.mode)
            self.context_data_frame = self.prophet_predicted_trade(self.context, period=abs(int(len(self.data)*0.7) - len(self.data)) if mode == 'test' else 30)
        else:
            self.context_data_frame = self.prophet_predicted_trade(period=abs(int(len(self.data)*0.7) - len(self.data)) if mode == 'test' else 30)
            self.garch_model = joblib.load(f"garch_{self.symbol}_{self.time_frame}_model.pkl")
            
        self.combined_trend_forecast, self.slope = self.combined_forecast_garch_prophet_rf(self.context_data_frame['close_forecast'], self.garch_model, self.data, mode=self.mode)
        self.recent_order_block = None 
        self.used_order_block = self.load_order_blocks(self.symbol)
        
        
        # Setup logging
        logging.basicConfig(filename='trading_log.log', level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def load_order_blocks(self,symbol):
        blocks = get_used_order_blocks(symbol=symbol)
        #print("[][]", blocks)
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

        #print(f"Best GARCH order: {best_garch_order}")
        return best_garch_order
    
    def arima_model_residual(self, data, best_order, price='Close', steps=30):
        arima_model = ARIMA(data[price], order=best_order).fit()
        arima_residuals = arima_model.resid
        arima_forecast = arima_model.forecast(steps=steps)
        # print(arima_forecast)
        return arima_residuals
    
    def garch_model_volatility(self, residuals, best_garch_order, period=30):
        # Fit the best GARCH model
        garch_model = arch_model(residuals, vol='Garch', p=best_garch_order[0], q=best_garch_order[1]).fit(disp='off')
        self.save_models(garch_model, 'garch', self.symbol, self.time_frame)
        return garch_model
    
    def get_symbol_data(self, symbol, duration, frame=M15):
        """Get historical price data for the symbol."""
        try:
            rates = mt5.copy_rates_from_pos(symbol, frame, 1, duration)
            if rates is None or len(rates) == 0:
                logging.error(f"Failed to get rates for {symbol}")
                return pd.DataFrame()
            
            rates_frame = pd.DataFrame(rates)
            rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
            rates_frame = rates_frame.rename(columns={'open':'Open', 'high':'High', 'low':'Low', 'close':'Close'})
            #print(rates_frame)
            return rates_frame
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()
    
    def prophet_analyze_data(self, data):
        prophet_df = data[['time', 'Open', 'High', 'Low', 'Close']]
        df_close = prophet_df[['time', 'Close']].rename(columns={'time':'ds', 'Close':'y'})
        df_open = prophet_df[['time', 'Open']].rename(columns={'time':'ds', 'Open':'y'})
        
        model_close = Prophet()
        model_close.fit(df_close)
        model_open = Prophet()
        model_open.fit(df_open)
        
        
        #print(f"Number of rows in future_close: {len(future_close)}")
        # Save the model
        self.save_models(model_close, 'close', self.symbol, self.time_frame)
        self.save_models(model_open, 'open', self.symbol, self.time_frame)
        
        context = {
            'model_close':model_close,
            'model_open':model_open,
        }
        return context

    def prophet_train_models(self, data, mode):
        if mode != "live" or self.retrain:
            train_len = int(len(data)*0.7)
            train = data[:train_len]
            period = abs(len(data) - train_len)
            return self.prophet_analyze_data(train)
        else:
            return self.prophet_analyze_data(data)
     
    def prophet_predicted_trade(self, context={}, period=30):
        if self.retrain:
            model_close = context['model_close']
            model_open = context['model_open']
        else:
            # Load the saved model
            model_close = joblib.load(f"close_{self.symbol}_{self.time_frame}_model.pkl")
            model_open = joblib.load(f"open_{self.symbol}_{self.time_frame}_model.pkl")
        
        
        if self.time_frame == M15:
            future_close = model_close.make_future_dataframe(periods=period, freq='15min', include_history=False)
            future_open = model_open.make_future_dataframe(periods=period, freq='15min', include_history=False)
        
        if self.time_frame == H1:
            future_close = model_close.make_future_dataframe(periods=period, freq='60min', include_history=False)
            future_open = model_open.make_future_dataframe(periods=period, freq='60min', include_history=False)
        
        close_forecast = model_close.predict(future_close)
        open_forecast = model_open.predict(future_open)

        context_data = {
            'time':open_forecast['ds'],
            'open_forecast':open_forecast['yhat'],
            'close_forecast':close_forecast['yhat'],
        }
        return pd.DataFrame(context_data)

    def save_models(self, model, model_name, symbol, timeframe):
        joblib.dump(model, f"{model_name}_{symbol}_{timeframe}_model.pkl")
        
    def classify_market_condition(self, data):
        # Prepare features and target
        # Prepare features
        features = data[['RSI', 'ATR', '20MA', '50MA', 
                        'BBands_Upper', 'BBands_Middle', 'BBands_Lower',
                        'MACD', 'MACD_SIGNAL', 'MACD_HIST']]
        
        # Add Price Action features
        features['Close_Diff'] = data['Close'].diff()
        features['High_Low_Range'] = data['High'] - data['Low']
        
        # Prepare target
        target = np.where(data['Close'].diff() > 0, 'Bullish', 'Bearish')  # Example target
        
        # Encode target labels
        le = LabelEncoder()
        target_encoded = le.fit_transform(target)
        
        # Train Random Forest classifier
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(features, target_encoded)
        self.save_models(model, 'RFC_market_condition', self.symbol, self.time_frame)
        return le
        
    def predict_market_condition(self,  mode='test'):
        # Train-test split
        if mode != "live" or self.retrain:
            train_len = int(len(self.data) * 0.7)
            train = self.data[:train_len]
            test = self.data[train_len:]
            
            # Encode labels from training data
            encoded = self.classify_market_condition(train)
        else:
            # Use entire dataset for live mode
            encoded = self.classify_market_condition(self.data)

        # Prepare features
        features = self.data[['RSI', 'ATR', '20MA', '50MA',
                            'BBands_Upper', 'BBands_Middle', 'BBands_Lower',
                            'MACD', 'MACD_SIGNAL', 'MACD_HIST']].copy()

        # Add price action features
        features['Close_Diff'] = self.data['Close'].diff()
        features['High_Low_Range'] = self.data['High'] - self.data['Low']

        # Drop NaNs from engineered features
        features.dropna(inplace=True)

        # Load model
        model = joblib.load(f"RFC_market_condition_{self.symbol}_{self.time_frame}_model.pkl")

        # Get last row for current prediction
        current_features = features.iloc[-1].values.reshape(1, -1)

        # Predict condition
        y_pred_label = model.predict(current_features)
        predicted_condition = encoded.inverse_transform(y_pred_label)[0]

        # Get probabilities
        probability = model.predict_proba(current_features)

        # Accuracy (test mode only)
        # if mode != 'live':
        #     aligned_y_true = y_true[-len(features):]  # Ensure y_true aligns with feature length
        #     y_pred_all = model.predict(features)
        #     accuracy = accuracy_score(aligned_y_true, y_pred_all)
        #     print(f"Test Accuracy: {accuracy:.2f}")
        #     print('Random ForestClassifier Report')
        #     print(classification_report(aligned_y_true, y_pred_all, target_names=encoded.classes_))

        print(f"Prediction Probability: {probability}")
        return predicted_condition, probability[0].max()
    
    def combined_forecast_garch_prophet_rf(self, prophet_forecast, garch_model, data, price='Close', period=30, mode='test'):
        # Combine features
        
        # Generate GARCH forecast
        if mode == 'test':
            garch_forecast = garch_model.forecast(horizon=int(len(self.data)*0.7))
        else:
            garch_forecast = garch_model.forecast(horizon=period)
        
        # Ensure garch_forecast.variance['h.01'] is not empty
        if garch_forecast.variance.empty:
            raise ValueError("GARCH forecast variance is empty. Check the GARCH model and input data.")
        
        # Expand garch_forecast_mean to match the shape of prophet_forecast_values
        garch_forecast_mean_expanded = np.repeat(garch_forecast.variance.iloc[-1, 0], prophet_forecast.values.shape[0])
        
        print(len(garch_forecast_mean_expanded))
        # print(garch_forecast_mean_expanded)
        print(prophet_forecast.values.shape)
        # print(prophet_forecast.values)
        
        # Combine features
        X = np.column_stack([prophet_forecast.values, garch_forecast_mean_expanded])
        y = data[price][-len(X):].values  # Align target with features
        
        # Check shapes
        print("X Shape:", X.shape,  len(X))
        print("y Shape:", y.shape, len(y))
        
        if mode != "live":
            # Train-test split
            split_index = len(X) - period
            print(split_index)
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]

            print("X Shape train-test:", X_train.shape, X_test.shape)
            print("y Shape train-test:", y_train.shape, y_test.shape)
            
            # Train Random Forest model
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            
            # Predict
            combined_forecast = model.predict(X_test)
            
            # Evaluation
            mae = mean_absolute_error(y_test, combined_forecast)
            mse = mean_squared_error(y_test, combined_forecast)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_test - combined_forecast) / y_test)) * 100
            accuracy = 100 - mape
            
            print('Random Forest Regression Evaluation Report:')
            print(f'MAE (Random Forest) values closer to zero are better: {mae}')
            print(f'MSE (Random Forest) values closer to zero are better: {mse}')
            print(f'RMSE (Random Forest) values closer to zero are better: {rmse}')
            print(f'MAPE (Random Forest) values closer to zero are better: {mean_absolute_percentage_error(y_test, combined_forecast)}, {mape}')
            print(f'Accuracy (Random Forest): {accuracy:.2f}%')
            print(f"Accuracy R2 Score (Random Forest) values closer to one are better: {r2_score(y_test, combined_forecast)}")
        else:
            # Train on full dataset
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            combined_forecast = model.predict(X)
        
        return pd.DataFrame({'combined_forecast': combined_forecast}), None

    def calculate_supertrend(self, period=7, multiplier=3):
        """
        Calculate the Supertrend indicator.

        Parameters:
            period (int): The period for calculating ATR. Default is 7.
            multiplier (int): The multiplier for calculating the bands. Default is 3.

        Returns:
            pd.DataFrame: The input DataFrame with additional columns for the Supertrend indicator.
        """
        # Validate input data
        required_columns = ['High', 'Low', 'Close', 'ATR']
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError(f"Input data must contain the following columns: {required_columns}")

        # Calculate HL2
        # Calculate HL2 (midpoint of High and Low)
        hl2 = (self.data['High'] + self.data['Low']) / 2

        # Calculate Upper and Lower Bands
        self.data['BasicUpperBand'] = hl2 + (multiplier * self.data['ATR'])
        self.data['BasicLowerBand'] = hl2 - (multiplier * self.data['ATR'])

        # Initialize Final Bands and Supertrend
        self.data['FinalUpperBand'] = self.data['BasicUpperBand']
        self.data['FinalLowerBand'] = self.data['BasicLowerBand']
        self.data['SuperTrend'] = np.nan

        # Calculate Final Bands and Supertrend
        for i in range(1, len(self.data)):
            # Adjust Final Upper Band
            if self.data['Close'].iloc[i-1] > self.data['FinalUpperBand'].iloc[i-1]:
                self.data['FinalUpperBand'].iloc[i] = min(self.data['BasicUpperBand'].iloc[i], self.data['FinalUpperBand'].iloc[i-1])
            else:
                self.data['FinalUpperBand'].iloc[i] = self.data['BasicUpperBand'].iloc[i]

            # Adjust Final Lower Band
            if self.data['Close'].iloc[i-1] < self.data['FinalLowerBand'].iloc[i-1]:
                self.data['FinalLowerBand'].iloc[i] = max(self.data['BasicLowerBand'].iloc[i], self.data['FinalLowerBand'].iloc[i-1])
            else:
                self.data['FinalLowerBand'].iloc[i] = self.data['BasicLowerBand'].iloc[i]

            # Calculate Supertrend
            if self.data['Close'].iloc[i] > self.data['FinalUpperBand'].iloc[i]:
                self.data['SuperTrend'].iloc[i] = self.data['FinalLowerBand'].iloc[i]
            elif self.data['Close'].iloc[i] < self.data['FinalLowerBand'].iloc[i]:
                self.data['SuperTrend'].iloc[i] = self.data['FinalUpperBand'].iloc[i]
            else:
                self.data['SuperTrend'].iloc[i] = self.data['SuperTrend'].iloc[i-1]

        
            return self.data['SuperTrend']

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
        #print('account_leverage', account_leverage)
        print('sl', sl)
        print('tp', tp)
        
        if order_type == BUY  or order_type == BUY_LIMIT:
            #print("Bull", order_type)
            price = mt5.symbol_info_tick(symbol).ask if order_type == BUY else entry_price
            sl = sl
            tp = tp
            print(price)
            print(price - sl)
            amt_risk = (price - sl) * lot_size / account_leverage
            print("amt_risk", amt_risk)
            
        else:
            #print("Bear", order_type)
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
        print(risk_margin < (account_balance * 0.03), amt_risk < (account_balance * 0.1), account_balance)
        if risk_margin < (account_balance * 0.03) and amt_risk < (account_balance * 0.1): # change to and later
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

    def calculate_indicators(self):
        """Calculate technical indicators."""
        try:
            # RSI
            self.data['RSI'] = ta.rsi(self.data['Close'], 14)

            # ATR
            self.data['ATR'] = ta.atr(self.data['High'], self.data['Low'], self.data['Close'], length=14)

            # Moving Averages
            self.data['20MA'] = ta.ma('sma', self.data['Close'], length=20)
            self.data['50MA'] = ta.ma('sma', self.data['Close'], length=50)

            # MACD
            self.data[['MACD', 'MACD_SIGNAL', 'MACD_HIST']] = ta.macd(self.data['Close'], 12, 26, 9)

            # Bollinger Bands
            bbands_df = ta.bbands(self.data['Close'], length=20, std=2)
            self.data['BBands_Upper'] = bbands_df['BBU_20_2.0']
            self.data['BBands_Middle'] = bbands_df['BBM_20_2.0']
            self.data['BBands_Lower'] = bbands_df['BBL_20_2.0']

        except Exception as e:
            logging.error(f"Error in calculate_indicators: {e}")

    def analyze_market(self):
        try:
            # Step 1: Calculate Technical Indicators
            self.calculate_indicators()

            # Step 2: Predict Market Condition
            market_condition, probability = self.predict_market_condition(self.mode)
            print(f"Predicted Market Condition: {market_condition}")     
            
            forecasted = self.combined_trend_forecast['combined_forecast']
            first_close = forecasted.iloc[-3]
            last_close = forecasted.iloc[-1]
            
            # Step 3: Market Sentiment
            print(f"symbol in news sentiment {self.symbol}")
            market_sentiment = self.news_sentiment_analysis(self.symbol)
            print(f"Market Sentiment: {market_sentiment}")
            
            # Check current market data
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
                    print(f"Identified Order Block: {order_block}")
        
        

            # Check if the order block is valid with the current market condition
            if latest_order_block != None:
                # print('latest_order_block')
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
                    print('valid order block')
                    candle_after_setup = order_block['Order Block Location'] + 2
                    print(candle_after_setup, 'candle_after block')
                    recent_data = self.data.iloc[candle_after_setup:]
                    
                    entry_price = None
                    stop_loss = None
                    take_profit = None
                    order_type = None
                    
                    if self.risk_level.lower() == 'high':
                        if order_block['Order Block'] == 'Bullish' or market_condition == 'Bullish' or market_sentiment == 'BUY':
                            revisits = recent_data[(recent_data['Low'] <= order_block['Order Block High'])]
                            print(f"revisits {revisits}")
                            if not len(revisits) > 0:
                                insert_order_block(symbol=self.symbol, 
                                        order_block_type=order_block['Order Block'], 
                                        high=order_block['Order Block High'], 
                                        low=order_block['Order Block Low'], 
                                        timestamp=pd.Timestamp(order_block['Order Block Time']).isoformat(),
                                        status="used")
                                
                                entry_price = order_block['Order Block High']
                                stop_loss = order_block['Order Block Low'] - atr
                                take_profit = entry_price + abs(entry_price - stop_loss) * self.reward_ratio
                                order_type = BUY_LIMIT
                                
                        elif order_block['Order Block'] == 'Bearish' or market_condition == 'Bearish' or market_sentiment == 'SELL':
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
                    elif self.risk_level.lower() == 'medium':
                        if order_block['Order Block'] == 'Bullish' and market_condition == 'Bullish' or market_sentiment == 'BUY':
                            revisits = recent_data[(recent_data['Low'] <= order_block['Order Block High'])]
                            print(f"revisits {revisits}")
                            if not len(revisits) > 0:
                                insert_order_block(symbol=self.symbol, 
                                        order_block_type=order_block['Order Block'], 
                                        high=order_block['Order Block High'], 
                                        low=order_block['Order Block Low'], 
                                        timestamp=pd.Timestamp(order_block['Order Block Time']).isoformat(),
                                        status="used")
                            
                                entry_price = order_block['Order Block High']
                                stop_loss = order_block['Order Block Low'] - atr
                                take_profit = entry_price + abs(entry_price - stop_loss) * self.reward_ratio
                                order_type = BUY_LIMIT
                            
                        elif order_block['Order Block'] == 'Bearish' and market_condition == 'Bearish' or market_sentiment == 'SELL':
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
                    else:
                        if order_block['Order Block'] == 'Bullish' and market_condition == 'Bullish' and market_sentiment == 'BUY':
                            revisits = recent_data[(recent_data['Low'] <= order_block['Order Block High'])]
                            print(f"revisits {revisits}")
                            if not len(revisits) > 0:
                                insert_order_block(symbol=self.symbol, 
                                        order_block_type=order_block['Order Block'], 
                                        high=order_block['Order Block High'], 
                                        low=order_block['Order Block Low'], 
                                        timestamp=pd.Timestamp(order_block['Order Block Time']).isoformat(),
                                        status="used")
                                
                                entry_price = order_block['Order Block High']
                                stop_loss = order_block['Order Block Low'] - atr
                                take_profit = entry_price + abs(entry_price - stop_loss) * self.reward_ratio
                                order_type = BUY_LIMIT
                                
                        elif order_block['Order Block'] == 'Bearish' and market_condition == 'Bearish' and market_sentiment == 'SELL':
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
                    
                    self.recent_order_block = latest_order_block
                    
                    if order_type:
                        self.send_order(
                                symbol=self.symbol,
                                lot_size=self.lot_size,
                                order_type=order_type,
                                entry_price=entry_price,
                                sl=stop_loss,
                                tp=take_profit,
                                comment=f"{market_condition} order block"
                            )
                    else:
                        print('No Order Type')
                else:
                    print("Expired Order Block")     
            else:
                print("No Order Block")
        except Exception as e:
            print(e)
            logging.error(f"Error in analyze_market: {e}")
                         
    def monitor_symbols(self, symbols, mode, time_frame, duration=880, risk_percentage=1.0, window_size=10, reward_ratio=2.2):
        """Monitor multiple symbols concurrently."""
        threads = [Thread(target=self.run_for_symbol, args=(symbol, mode, time_frame, duration, risk_percentage, window_size, reward_ratio)) for symbol in symbols]
        for thread in threads:
            thread.start()
            logging.info(f"Thread Started: {thread}")

        for thread in threads:
            thread.join()

    def news_sentiment_analysis(self, symbol):
        """Perform sentiment analysis on the data."""
        # You can use libraries like TextBlob or VADER for sentiment analysis
        try:
            df = pd.read_csv('forex_factory_calendar.csv')
        except:
            df = pd.DataFrame(get_news_data())
        
        symbol_data = df[df['Currency'].isin([symbol[:3], symbol[3:]])]
        symbol_data = symbol_data[symbol_data['Impact'].isin(['High Impact Expected', 'Medium Impact Expected'])]  # Filter by impact level
        now_time = datetime.datetime.now().time()
        symbol_data['Time'] = pd.to_datetime(symbol_data['Time'], format='mixed').dt.time
        symbol_data = symbol_data[symbol_data['Time'] >= now_time]
        
        print(symbol_data)
        
        if symbol_data.empty:
            return "NEUTRAL"

        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = []

        for event in symbol_data['Event']:
            score = analyzer.polarity_scores(event)['compound']
            if abs(score) > 0.2:  # Consider meaningful sentiment
                sentiment_scores.append(score)

        if not sentiment_scores:
            return "NEUTRAL"

        try:
            avg_score = sum(sentiment_scores) / len(sentiment_scores) if len(sentiment_scores) > 1 else sentiment_scores[0]
        except ZeroDivisionError:
            avg_score = 0

        if avg_score > 0.1:
            return "BUY"
        elif avg_score < -0.1:
            return "SELL"
        else:
            return "NEUTRAL"
        
        
    def run_for_symbol(self, symbol, mode, time_frame, duration, risk_percentage, window_size, reward_ratio):
        """Run market analysis for a specific symbol."""
        model = BlueKiteModel(symbol, mode, time_frame, duration, risk_percentage, window_size, reward_ratio)
        model.analyze_market()
        logging.info(f"Analying Market for {symbol}")
    


# BlueKite 11.5.0
# Define active and inactive times
ACTIVE_START_TIME = "09:30"
ACTIVE_END_TIME = "16:30"


# connect to MetaTrader 5
if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()

def start_bot():
    symbols = ['XAUUSD', 'EURUSD', 'EURGBP', 'GBPJPY', 'GBPUSD', 'APPLE', 'GOOGLE', 'BTCUSD']

    model = BlueKiteModel(symbol='EURUSD', mode='test', time_frame=M15, retrain=True)
    model.analyze_market()
    # model.monitor_symbols(symbols,  mode='live', time_frame=M15)



def start_job():
    # Ensure the job isn’t already scheduled
    if not any(job.job_func == start_bot for job in schedule.jobs):
        schedule.every(15).minutes.do(start_bot)
        print("Scheduled start_bot job")

def stop_job():
    for job in list(schedule.jobs):
        if job.job_func == start_bot:
            schedule.cancel_job(job)
            print("Trading job stopped")

def check_time_and_run():
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M")
    current_day = now.strftime("%A")

    if current_day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        if ACTIVE_START_TIME <= current_time <= ACTIVE_END_TIME:
            start_job()
        else:
            stop_job()
    else:
        stop_job()

# Schedule daily news
schedule.every().day.at("08:00").do(get_news_data)

# Schedule the check every minute
schedule.every(1).minutes.do(check_time_and_run)

# Run loop
# while True:
#     schedule.run_pending()
#     time.sleep(1)

start_bot()
