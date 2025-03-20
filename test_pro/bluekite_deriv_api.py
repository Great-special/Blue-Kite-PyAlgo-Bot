import numpy as np
import pandas as pd
import logging
import joblib
import json
import asyncio
import schedule
import datetime
import time
import pmdarima as pm
import warnings
import pandas_ta as ta

from db import insert_order_block, get_used_order_blocks
from deriv_api import DerivAPI
from threading import Thread
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import het_arch
from arch import arch_model
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Automatic order selection for ARIMA
warnings.filterwarnings("ignore")

api_token = 'Ehh3Uh4LIaLt2m0'
app_id = 69587

# Order Type Constants (replacing MT5 constants)
BUY = 'BUY'
BUY_LIMIT = 'BUY_LIMIT'
SELL = 'SELL'
SELL_LIMIT = 'SELL_LIMIT'

# TimeFrame Constants (replacing MT5 constants)
D1 = 'D1'  # 1 day
H1 = 'H1'  # 1 hour
H4 = 'H4'  # 4 hours
M30 = 'M30'  # 30 minutes
M15 = 'M15'  # 15 minutes
M5 = 'M5'    # 5 minutes

# Mapping time frames to Deriv intervals
TIMEFRAME_MAP = {
    D1: 86400,    # seconds in a day
    H4: 14400,    # seconds in 4 hours
    H1: 3600,     # seconds in an hour
    M30: 1800,    # seconds in 30 minutes
    M15: 900,     # seconds in 15 minutes
    M5: 300,      # seconds in 5 minutes
}

class BlueKiteModel:
    
    def __init__(self, symbol, data=None, mode='test', time_frame=H1, duration=880, risk_percentage=1.0, window_size=10, reward_ratio=1.8, retrain=True):
        self.mode = mode
        self.symbol = symbol
        self.time_frame = time_frame
        self.retrain = retrain
        self.data = data
        self.lot_size = 0.01  # Default lot size
        self.risk_percentage = risk_percentage
        self.window_size = window_size
        self.reward_ratio = reward_ratio
        self.prev_high = None
        self.prev_low = None
        self.new_high = None
        self.new_low = None
        self.recent_order_block = None 
        self.used_order_block = self.load_order_blocks(self.symbol)
        if self.retrain:
            # Perform initial analysis and model training
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
       
        
        # Setup logging
        logging.basicConfig(filename='trading_log.log', level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
    
    def load_order_blocks(self, symbol):
        blocks = get_used_order_blocks(symbol=symbol)
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
        best_garch_order = (1, 1)  # Default order
        
        for p in range(1, 4):
            for q in range(1, 4):
                try:
                    garch_temp = arch_model(residuals, vol='Garch', p=p, q=q).fit(disp='off')
                    if garch_temp.aic < best_garch_aic:
                        best_garch_aic = garch_temp.aic
                        best_garch_order = (p, q)
                except:
                    continue

        logging.info(f"Best GARCH order: {best_garch_order}")
        return best_garch_order
    
    def arima_model_residual(self, data, best_order, price='Close', steps=30):
        arima_model = ARIMA(data[price], order=best_order).fit()
        arima_residuals = arima_model.resid
        arima_forecast = arima_model.forecast(steps=steps)
        return arima_residuals
    
    def garch_model_volatility(self, residuals, best_garch_order, period=30):
        # Fit the best GARCH model
        garch_model = arch_model(residuals, vol='Garch', p=best_garch_order[0], q=best_garch_order[1]).fit(disp='off')
        self.save_models(garch_model, 'garch', self.symbol, self.time_frame)
        return garch_model
    
    def prophet_analyze_data(self, data):
        prophet_df = data[['time', 'Open', 'High', 'Low', 'Close']]
        df_close = prophet_df[['time', 'Close']].rename(columns={'time':'ds', 'Close':'y'})
        df_open = prophet_df[['time', 'Open']].rename(columns={'time':'ds', 'Open':'y'})
        
        model_close = Prophet()
        model_close.fit(df_close)
        model_open = Prophet()
        model_open.fit(df_open)
        
        # Save the model
        self.save_models(model_close, 'close', self.symbol, self.time_frame)
        self.save_models(model_open, 'open', self.symbol, self.time_frame)
        
        context = {
            'model_close': model_close,
            'model_open': model_open,
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
        elif self.time_frame == H1:
            future_close = model_close.make_future_dataframe(periods=period, freq='60min', include_history=False)
            future_open = model_open.make_future_dataframe(periods=period, freq='60min', include_history=False)
        else:
            # Default to hourly if time frame not recognized
            future_close = model_close.make_future_dataframe(periods=period, freq='60min', include_history=False)
            future_open = model_open.make_future_dataframe(periods=period, freq='60min', include_history=False)
        
        close_forecast = model_close.predict(future_close)
        open_forecast = model_open.predict(future_open)

        context_data = {
            'time': open_forecast['ds'],
            'open_forecast': open_forecast['yhat'],
            'close_forecast': close_forecast['yhat'],
        }
        return pd.DataFrame(context_data)

    def save_models(self, model, model_name, symbol, timeframe):
        joblib.dump(model, f"{model_name}_{symbol}_{timeframe}_model.pkl")
        
    def classify_market_condition(self, data):
        # Prepare features and target
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
        model = RandomForestClassifier(n_estimators=500, random_state=42)
        model.fit(features, target_encoded)
        self.save_models(model, 'RFC_market_condition', self.symbol, self.time_frame)
        return le
        
    def predict_market_condition(self, data):
        encoded = self.classify_market_condition(self.data)
        
        # Predict current market condition
        features = data[['RSI', 'ATR', '20MA', '50MA', 
                        'BBands_Upper', 'BBands_Middle', 'BBands_Lower',
                        'MACD', 'MACD_SIGNAL', 'MACD_HIST']]
        
        # Add Price Action features
        features['Close_Diff'] = data['Close'].diff()
        features['High_Low_Range'] = data['High'] - data['Low']
        
        model = joblib.load(f"RFC_market_condition_{self.symbol}_{self.time_frame}_model.pkl")
        current_features = features.iloc[-1].values.reshape(1, -1)
        predicted_condition = encoded.inverse_transform(model.predict(current_features))[0]
        probability = model.predict_proba(current_features)
        logging.info(f"Market condition: {predicted_condition}, Probability: {probability[0].max()}")
        return predicted_condition, probability[0].max()
    
    def combined_forecast_garch_prophet_rf(self, prophet_forecast, garch_model, data, price='Close', period=30, mode='test'):
        # Generate GARCH forecast
        if mode == 'test':
            garch_forecast = garch_model.forecast(horizon=int(len(self.data)*0.7))
        else:
            garch_forecast = garch_model.forecast(horizon=period)
        
        # Ensure garch_forecast.variance is not empty
        if garch_forecast.variance.empty:
            raise ValueError("GARCH forecast variance is empty. Check the GARCH model and input data.")
        
        # Expand garch_forecast_mean to match the shape of prophet_forecast_values
        garch_forecast_mean_expanded = np.repeat(garch_forecast.variance.iloc[-1, 0], prophet_forecast.values.shape[0])
        
        logging.info(f"GARCH forecast shape: {len(garch_forecast_mean_expanded)}")
        logging.info(f"Prophet forecast shape: {prophet_forecast.values.shape}")
        
        # Combine features
        X = np.column_stack([prophet_forecast.values, garch_forecast_mean_expanded])
        y = data[price][-len(X):].values  # Align target with features
        
        logging.info(f"Combined features shape: {X.shape}")
        logging.info(f"Target shape: {y.shape}")
        
        if mode != "live":
            # Train-test split
            split_index = len(X) - period
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]

            logging.info(f"Training features shape: {X_train.shape}, Test features shape: {X_test.shape}")
            
            # Train Random Forest model
            model = RandomForestRegressor(n_estimators=500, random_state=42)
            model.fit(X_train, y_train)
            
            # Predict
            combined_forecast = model.predict(X_test)
            
            # Evaluation
            mae = mean_absolute_error(y_test, combined_forecast)
            mse = mean_squared_error(y_test, combined_forecast)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_test - combined_forecast) / y_test)) * 100
            accuracy = 100 - mape
            
            logging.info(f'MAE (Random Forest): {mae}')
            logging.info(f'MSE (Random Forest): {mse}')
            logging.info(f'RMSE (Random Forest): {rmse}')
            logging.info(f'Accuracy (Random Forest): {accuracy:.2f}%')
        else:
            # Train on full dataset
            model = RandomForestRegressor(n_estimators=500, random_state=42)
            model.fit(X, y)
            combined_forecast = model.predict(X)
        
        return pd.DataFrame({'combined_forecast': combined_forecast}), None
    
    async def send_order(self, symbol, amount, contract_type, entry_price, sl, tp, comment):
        """Send a trading order using Deriv API"""
        try:
            # Risk Management
            account_balance = self.account_info.get('balance', 0)
            account_leverage = self.account_info.get('leverage', 100)
            
            # Calculate risk
            if contract_type in [BUY, BUY_LIMIT]:
                price = entry_price
                amt_risk = (price - sl) * amount / account_leverage
                logging.info(f"Buy order - Price: {price}, SL: {sl}, TP: {tp}, Risk: {amt_risk}")
            else:
                price = entry_price
                amt_risk = (sl - price) * amount / account_leverage
                logging.info(f"Sell order - Price: {price}, SL: {sl}, TP: {tp}, Risk: {amt_risk}")

            risk_margin = (price * amount) / account_leverage
            logging.info(f"Risk margin: {risk_margin}")
            
            # Check if risk is acceptable
            if risk_margin < (account_balance * 0.03) and amt_risk < (account_balance * 0.1):
                # Prepare trading parameters
                duration_unit = "d"  # day
                duration = 1  # 1 day
                
                # Convert contract type to Deriv API format
                if contract_type in [BUY, BUY_LIMIT]:
                    direction = "CALL"
                else:
                    direction = "PUT"
                
                # For spot forex or other trading
                proposal_request = {
                    "proposal": 1,
                    "amount": amount,
                    "basis": "stake",
                    "contract_type": direction,
                    "currency": "USD",
                    "duration": duration,
                    "duration_unit": duration_unit,
                    "symbol": symbol,
                    "barrier": price
                }
                
                # Get proposal first
                proposal = await self.api.send(proposal_request)
                
                if 'proposal' in proposal and 'id' in proposal['proposal']:
                    # Buy the contract
                    buy_request = {
                        "buy": proposal['proposal']['id'],
                        "price": amount
                    }
                    
                    result = await self.api.send(buy_request)
                    
                    if 'buy' in result:
                        contract_id = result['buy']['contract_id']
                        logging.info(f"Order placed successfully. Contract ID: {contract_id}")
                        return {"success": True, "contract_id": contract_id}
                    else:
                        logging.error(f"Order placement failed: {result}")
                        return {"success": False, "error": "Failed to place order"}
                else:
                    logging.error(f"Failed to get proposal: {proposal}")
                    return {"success": False, "error": "Failed to get proposal"}
            else:
                logging.warning("Risk management criteria not met. Order not placed.")
                return {"success": False, "error": "Risk management criteria not met"}
                
        except Exception as e:
            logging.error(f"Error sending order: {e}")
            return {"success": False, "error": str(e)}

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
            print('HERE-Analyze Market')
            # Step 1: Calculate Technical Indicators
            self.calculate_indicators()

            # Step 2: Predict Market Condition
            market_condition, probability = self.predict_market_condition(self.data)
            logging.info(f"Predicted Market Condition: {market_condition}")     
            
            forecasted = self.combined_trend_forecast['combined_forecast']
            first_close = forecasted.iloc[-3]
            last_close = forecasted.iloc[-1]
            
            # Check current market data
            current = self.data.iloc[-1]
            logging.info(f"Current market data: {current}")
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
            
            # Check if the order block is valid with the current market condition
            if latest_order_block is not None:
                order_block, atr = latest_order_block
                if self.used_order_block is not None:
                    for block in self.used_order_block:
                        if block[1] == self.symbol and block[2] == order_block['Order Block'] and order_block['Order Block High'] == block[3] and order_block['Order Block Low'] == block[4]:
                            UsedOrderBlock = True
                
                # Checking the order block duration
                order_duration = abs(current['time'] - order_block['Order Block Time'])
                
                # Extracting difference in terms of days, hours, and minutes
                days = order_duration.days
                hours, remainder = divmod(order_duration.seconds, 3600)
                minutes, _ = divmod(remainder, 60)
                logging.info(f"Order block age: {days} days, {hours} hours, {minutes} minutes")            

                if minutes <= 60 and hours <= 1 and not UsedOrderBlock:
                    logging.info('Valid order block found')
                    candle_after_setup = order_block['Order Block Location'] + 2
                    logging.info(f"Candle after block: {candle_after_setup}")
                    recent_data = self.data.iloc[candle_after_setup:]
                    
                    entry_price = None
                    stop_loss = None
                    take_profit = None
                    order_type = None
                    
                    if order_block['Order Block'] == 'Bullish' and market_condition == 'Bullish':
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
                    
                    elif order_block['Order Block'] == 'Bearish' and market_condition == 'Bearish':
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
                    print('here')
                    if order_type:
                        #  self.send_order(
                        #         symbol=self.symbol,
                        #         lot_size=self.lot_size,
                        #         order_type=order_type,
                        #         entry_price=entry_price,
                        #         sl=stop_loss,
                        #         tp=take_profit,
                        #         comment=f"{market_condition} order block"
                        #     )
                        print('returned')
                        return {
                            'symbol':self.symbol,
                            'lot_size':self.lot_size,
                            'order_type':order_type,
                            'entry_price':entry_price,
                            'sl':stop_loss,
                            'tp':take_profit,
                            'comment':f"{market_condition} order block"
                        }
                    else:
                        return {}
                else:
                    print("Expired Order Block")
            else:
                print("No Order Block")
        except Exception as e:
            logging.error(f"Error in analyze_market: {e}")
            

async def blueMain():
    # Initialize the Deriv API connection
    api = DerivAPI(app_id=app_id)
    authorize = await api.authorize(api_token)
    print(authorize.keys())
    
    symbols = await get_symbols(api)
    symbol = 'frxEURNZD'
    for sym in symbols:
        if sym['symbol'] == symbol:
            lot_size = sym['lot_size']
            print(lot_size)
            break
    # Get account info
    account_info = await get_account_info(api)
    
    # Get historical data
    data = await get_symbol_data(symbol, api=api, time_frame=M15, duration=880)
    
    model = BlueKiteModel(symbol=symbol, data=data, mode='live', time_frame=M15, retrain=False)
    proposal = model.analyze_market()
    
    # Risk Management
    account_balance = account_info.get('balance', 0)
    account_leverage = account_info.get('leverage', 100)
    
    # Calculate risk
    try:
        order_type = proposal.get('order_type')
        entry_price = proposal.get('entry_price')
        sl = proposal.get('sl')
        tp = proposal.get('tp')
        
        if order_type in [BUY, BUY_LIMIT]:
            price = entry_price
            amt_risk = (price - sl) * lot_size / account_leverage
            print(f"Buy order - Price: {price}, SL: {sl}, TP: {tp}, Risk: {amt_risk}")
        else:
            price = entry_price
            amt_risk = (sl - price) * lot_size / account_leverage
            print(f"Sell order - Price: {price}, SL: {sl}, TP: {tp}, Risk: {amt_risk}")

        risk_margin = (price * lot_size) / account_leverage
        print(f"Risk margin: {risk_margin}")
        
        # Check if risk is acceptable
        if risk_margin < (account_balance * 0.03) and amt_risk < (account_balance * 0.1):
            # Prepare trading parameters
            duration_unit = "d"  # day
            duration = 1  # 1 day
            
            # Convert contract type to Deriv API format
    except:
        pass

async def get_account_info(api):
    """Get account information from Deriv API"""
    try:
        # response = await api.send({'get_account_status': 1})
        # print(response, 'account status')
        
        balance_request = await api.balance()
        """
        If you want to retrieve all available trading instruments for a specific asset (e.g., Gold), you would use the asset parameter in the API request.
        If you want to retrieve market data or place a trade for a specific instrument (e.g., R_100), you would use the symbol parameter.
        """
        # assets = await api.asset_index({"asset_index": 1})
        # print(assets.get('asset_index'))
        
        account_info = {
            'balance': balance_request.get('balance', {}).get('balance', 0),
            'currency': balance_request.get('balance', {}).get('currency', 'USD'),
            'leverage': 100,  # Default leverage since Deriv API might not provide this directly
            # 'account_type': response.get('get_account_status', {})
        }
        # print(account_info)
        return account_info
    except Exception as e:
        logging.error(f"Error getting account info: {e}")
        return {'balance': 0, 'currency': 'USD', 'leverage': 100, 'account_type': 'unknown'}


async def get_symbols(api):
        active_symbols = await api.active_symbols({"active_symbols": "full"})
        print('active symbols',  len(active_symbols.get('active_symbols')))
        symbols = [
            {'market_type':syn_row.get('market'), 
             'symbol':syn_row.get('symbol'), 
             'lot_size':syn_row.get('pip'), 
             'symbol_display_name':syn_row.get('display_name'), 
             'is_open':syn_row.get('exchange_is_open'), 
             'is_suspended':syn_row.get('is_trading_suspended')}
            for syn_row in active_symbols.get('active_symbols')
            ]
        return symbols



async def get_symbol_data(symbol, time_frame, api, duration=880):
    """Get historical price data for the symbol using Deriv API."""
    try:
        # Convert time_frame to granularity in seconds
        granularity = TIMEFRAME_MAP.get(time_frame, 3600)  # Default to 1 hour
        
        # Calculate start time based on duration and granularity
        end_epoch = int(time.time())
        start_epoch = end_epoch - (granularity * duration)
        
        # Request candles from Deriv API
        candles_request = {
            "ticks_history": symbol,
            "granularity": granularity,
            "start": start_epoch,
            "end": end_epoch,
            "style": "candles"
        }
        
        response = await api.send(candles_request)
        
        if 'candles' not in response or not response['candles']:
            logging.error(f"Failed to get candles for {symbol}")
            return pd.DataFrame()
        
        # Extract candle data
        candles = response['candles']
        data = []
        
        for candle in candles:
            data.append({
                'time': datetime.datetime.fromtimestamp(candle['epoch']),
                'Open': float(candle['open']),
                'High': float(candle['high']),
                'Low': float(candle['low']),
                'Close': float(candle['close'])
            })
        
        df = pd.DataFrame(data)
        logging.info(f"Retrieved {len(df)} candles for {symbol}")
        return df
    
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()




asyncio.run(blueMain())