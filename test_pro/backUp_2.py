import numpy as np
import itertools
import schedule
import time
import pmdarima as pm
import warnings
import pandas as pd
import pandas_ta as ta

import MetaTrader5 as mt5
from datetime import datetime
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import het_arch
from arch import arch_model
from prophet import Prophet
from findpeaks import findpeaks
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression

# Automatic order selection for ARIMA
warnings.filterwarnings("ignore")


# connect to MetaTrader 5
if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()
 
# request connection status and parameters
# print(mt5.terminal_info())

# get data on MetaTrader 5 version
# print(mt5.version())
# print("-+-"*20)
# print(mt5.account_info()._asdict())

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

TRADEABLE = ['XAUUSD', 'EURUSD', 'GBPJPY', 'NZDCHF', 'APPLE', 'GOOGLE', 'BTCUSD']




def auto_arima_order(data, price='close'):
    p = d = q = range(0, 6)
    pdq = list(itertools.product(p, d, q))

    best_aic = float("inf")
    best_order = None

    for order in pdq:
        try:
            arima_temp = ARIMA(data[price], order=order).fit()
            if arima_temp.aic < best_aic:
                best_aic = arima_temp.aic
                best_order = order
        except:
            continue

    print(f"Best ARIMA order: {best_order}")
    return best_order


def auto_garch_order(residuals):
    best_garch_aic = float("inf")
    best_garch_order = None

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



def connect_account(account=None, password=None, server=None):
    if account != None and password != None:
        authorized = mt5.login(account, password)
        
        if authorized:
            print("-+-"*20)
            print(mt5.account_info()._asdict())
        else:
            print('*'*20)
            print('failed to connect at account {}'.format(account))


def get_symbols():
    symbols = mt5.symbols_get()
    print('Symbols: ', len(symbols))
    list_symbols_info = []
    count=0
    # display the first five ones
    for sym in symbols:
        count+=1
        
        #display symbol properties / info
        symbol_info = mt5.symbol_info(sym.name)
        
        if symbol_info!=None:
            symbol_name = symbol_info.name
            symbol_min_lotsize = symbol_info.volume_min  
            symbol_max_lotsize = symbol_info.volume_max
            symbol_tick_value = symbol_info.trade_tick_value   
            
            list_symbols_info.append({
                'symbolName': symbol_name,
                'symbolMinLot': symbol_min_lotsize,
                'symbolMaxLot': symbol_max_lotsize,
                'symbolTickValue' : symbol_tick_value
            })
    
    print(list_symbols_info)


def get_symbol_data(symbol, duration, frame=H1):
    """Getting the symbol info and the price or historical price"""
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info!=None:
        symbol_name = symbol_info.name
        symbol_min_lotsize = symbol_info.volume_min  
        symbol_max_lotsize = symbol_info.volume_max
        symbol_tick_value = symbol_info.trade_tick_value
    
    rates = mt5.copy_rates_from_pos(symbol, frame, 1, duration)
    # print(symbol_name)
    rates_frame = pd.DataFrame(rates)
    try:
        rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
    except:
        print(rates_frame.columns)
    # print(rates_frame.tail())
    # rates_frame.to_excel(f'{symbol}H1.xlsx')
    
    
    symbol_info_frame = {
        'symbol name':symbol, 
        'symbol min lot size':symbol_min_lotsize,
        'symbol max lot size':symbol_max_lotsize,
        'symbol tick value':symbol_tick_value
    }

    
    return {'rates_frame':rates_frame, 'symbol_info_frame':symbol_info_frame}
    

def arima_model_residual(data, best_order, price='close', steps=30):
    arima_model = ARIMA(data[price], order=best_order).fit()
    arima_residuals = arima_model.resid
    arima_forecast = arima_model.forecast(steps=steps)
    # print(arima_forecast)
    return arima_residuals


def garch_model_volatility(residuals, best_garch_order, horizon=30):
    # Fit the best GARCH model
    garch_model = arch_model(residuals, vol='Garch', p=best_garch_order[0], q=best_garch_order[1]).fit(disp='off')
    garch_forecast = garch_model.forecast(horizon=horizon)
    return garch_forecast


def analyze_data(data, period=30):
    prophet_df = data[['time', 'open', 'high', 'low', 'close']]
    df_close = prophet_df[['time', 'close']].rename(columns={'time':'ds', 'close':'y'})
    df_open = prophet_df[['time', 'open']].rename(columns={'time':'ds', 'open':'y'})
    
    model_close = Prophet()
    model_close.fit(df_close)
    model_open = Prophet()
    model_open.fit(df_open)
    
    future_close = model_close.make_future_dataframe(periods=period, freq='15min', include_history=False)
    future_open = model_open.make_future_dataframe(periods=period, freq='15min', include_history=False)
    
    print(f"Number of rows in future_close: {len(future_close)}")
    
    context = {
        'model_close':model_close,
        'future_close':future_close,
        'model_open':model_open,
        'future_open':future_open,
    }
    return context


def train_models(data, period=30, mode="test"):
    if mode != "live":
        train_len = int(len(data)*0.7)
        train = data[:train_len]
        period = abs(len(data) - train_len)
        return analyze_data(train, period)
    else:
        return analyze_data(data, period)
        


def predict_trade(context):
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
    return context_data
    

def test_models(data, context):
    test_len = int(len(data)*0.7)
    test = data[test_len:]
    
    test = test.reset_index(drop=True)
    context_data = context
    open_forecast = context_data['open_forecast']
    close_forecast = context_data['close_forecast']
    
    evaluation = pd.DataFrame(
        {
            'actual_date':test['time'],
            'forecast_date':context_data['time'],
            
            'actual_open':test['open'],
            'predicted_open':open_forecast,
            '%Diff_open':((open_forecast - test['open'])/test['open']) * 100,
            
            'actual_close':test['close'],
            'predicted_close':close_forecast,
            '%Diff_close':((close_forecast - test['close'])/test['close']) * 100,
            'direction': open_forecast  < close_forecast
            
        }
    )
    
    evaluation['direction'] = evaluation['direction'].map({True:"Up", False:"Down"})
    
    print(evaluation)
    print(evaluation.tail(30))
    return evaluation


def combine_forecast(prophet_forecast, garch_forecast, data, price='close', horizon=30, mode='test'):
    if not mode == "live":
        # Check columns of garch_forecast
        # print("Columns in garch_forecast:")
        # print(garch_forecast.variance.tail())
        # print(garch_forecast.mean.columns)
        # print(garch_forecast.mean.tail())
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
        print(X_full)
            
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
        
        
        return pd.DataFrame({'combined_forecast':combined_forecast_ml})
    else:
        # Expand garch_forecast_mean to match the shape of prophet_forecast_values
        garch_forecast_mean_expanded = np.repeat(garch_forecast.variance['h.01'], prophet_forecast.values.shape[0])

        # Verify shapes
        print("Shape of prophet_forecast.values:", prophet_forecast.values.shape)
        print("Shape of garch_forecast.mean (expanded):", garch_forecast_mean_expanded.shape)

        # Combine arrays
        X = np.vstack([prophet_forecast.values, garch_forecast_mean_expanded]).T
        print("Combined Forecast Shape:", X.shape)
        print(X)
        
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
        return pd.DataFrame({'combined_forecast':combined_forecast_ml})



def accuracy(evaluation_data):
    mae_close = mean_absolute_error(evaluation_data['actual_close'], evaluation_data['predicted_close'])
    mse_close = mean_squared_error(evaluation_data['actual_close'], evaluation_data['predicted_close'])
    mape_close = mean_absolute_percentage_error(evaluation_data['actual_close'], evaluation_data['predicted_close'])
    accuracy_close = 100 * (1 -mape_close)
    
    mae_open = mean_absolute_error(evaluation_data['actual_open'], evaluation_data['predicted_open'])
    mse_open = mean_squared_error(evaluation_data['actual_open'], evaluation_data['predicted_open'])
    mape_open = mean_absolute_percentage_error(evaluation_data['actual_open'], evaluation_data['predicted_open'])
    accuracy_open = 100 * (1 -mape_open)
    
    print('-'*30) 
    print(f'Close Mean Absolute Error (MAE): {mae_close:.2f}')
    print(f'Close Mean Squared Error (MSE): {mse_close:.2f}')
    print(f'Close Accuracy: {accuracy_close:.2f}')
    print('-'*30)
    print(f'Open Mean Absolute Error (MAE): {mae_open:.2f}')
    print(f'Open Mean Squared Error (MSE): {mse_open:.2f}')
    print(f'Open Accuracy: {accuracy_open:.2f}')

    

def margin_checker(symbol, action, lot_size, price):
    if symbol:
        margin = mt5.order_calc_margin(action, symbol, lot_size, price)
        return margin
    else:
        print('error order margin checker failed!')


def profit_checker(action, symbol, lot_size, open_price, close_price):
    # estimate profit for buying and selling
    # get account currency
    account_currency=mt5.account_info().currency
    if symbol or action:
        symbol_tick=mt5.symbol_info_tick(symbol)
        ask=symbol_tick.ask
        bid=symbol_tick.bid
        if open_price < close_price and action == mt5.ORDER_TYPE_BUY and open_price >= ask:
            buy_profit=mt5.order_calc_profit(action, symbol, lot_size, open_price, close_price)
            if buy_profit!=None:
                print("   buy {} {} lot: profit on {} points => {} {}".format(symbol, lot_size, abs(open_price - close_price), buy_profit, account_currency))
            else:
                print("order_calc_profit(ORDER_TYPE_BUY) failed, error code =",mt5.last_error())
        else:
            sell_profit=mt5.order_calc_profit(action, symbol, lot_size, open_price, close_price)
            if sell_profit!=None:
                print("   sell {} {} lots: profit on {} points => {} {}".format(symbol,lot_size, abs(open_price - close_price), sell_profit,account_currency))
            else:
                print("order_calc_profit(ORDER_TYPE_SELL) failed, error code =",mt5.last_error())
            print()


def check_order_eligibility(request):
    result = mt5.order_check(request)
    print(result)
    
    return result

def create_request(symbol, lot_size, order_type, price, sl, tp):
    # Retrieve symbol information
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Symbol {symbol} not found")
        return None
    
    deviation = 5

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": deviation,
        "magic": 115,  # Magic number should be an integer
        "comment": "python script open",
        "type_time": mt5.ORDER_TIME_GTC,
        # "type_filling": mt5.ORDER_FILLING_RETURN,
    }
    return request

def send_order(symbol, lot_size, order_type, sl, tp):
    # Risk Management
    account_info = mt5.account_info() 
    account_leverage = account_info.leverage
    account_balance = account_info.balance
    print('account_leverage', account_leverage)
    
    if order_type == BUY :
        order_type = order_type
        price = mt5.symbol_info_tick(symbol).ask 
        sl = sl
        tp = tp
        amt_risk = (price - sl) * lot_size / account_leverage
        print("amt_risk", amt_risk)
        
    else:
        order_type = order_type
        price = mt5.symbol_info_tick(symbol).bid
        sl = sl
        tp = tp
        amt_risk = (sl - price) * lot_size / account_leverage
        print("amt_risk", amt_risk)

    risk_margin = (price * lot_size) / account_leverage
    print("risk_margin", risk_margin)
    
    if risk_margin < (account_balance * 0.02) and amt_risk < (account_balance * 0.05): # change to and later
        # Checking for open positions and number
        position_total = mt5.positions_total()
        position_symbol = mt5.positions_get(symbol=symbol)
        
        if position_total != 0 and len(position_symbol) != 0:
            print(f"Positions already open for symbol: {symbol}. Order not placed.")
            
        
        request = create_request(
            symbol=symbol, lot_size=lot_size, order_type=order_type, 
            price=price, sl=sl, tp=tp
        )
        
        # res = check_order_eligibility(request)
        
        if request is None:
            print("Failed to create order request")
            
        # Send a trading request
        result = mt5.order_send(request)
        
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
        
    
    

def start_bot():
    # mode can be 'live' or 'test'
    mode = 'test'
    start_time = time.perf_counter()
    print(f'Start Time: {time.strftime("%Y-%m-%d %H:%M")}')
    for i in TRADEABLE:
        symbol_name = i
        
        print(" - " * 25, symbol_name, " - " * 25)
        
        # Fetch symbol data
        res = get_symbol_data(symbol_name, 5040, M15)
        data = res['rates_frame']
        symbol_info = res['symbol_info_frame']
        
        print(data)
        # Initialize findpeaks
        fp = findpeaks(method='topology', lookahead=5, whitelist=['peak', 'valley'])
        print("Findpeaks initialized.")
        
        if mode == "live":
            # ARIMA model fitting
            best_arima_order = pm.auto_arima(data['close'], seasonal=False, stepwise=True, trace=True).order
            arima_residual = arima_model_residual(data, best_arima_order)
            
            # GARCH model fitting
            best_garch_order = auto_garch_order(arima_residual)
            garch_forecast = garch_model_volatility(arima_residual, best_garch_order)
            
            # Prophet model training and forecasting
            context = train_models(data, mode)
            context_data = predict_trade(context)
            context_frame = pd.DataFrame(context_data)
            context_frame['Trend'] = context_frame['open_forecast'] < context_frame['close_forecast']
            context_frame['Trend'] = context_frame['Trend'].map({True: "Up", False: "Down"})
            
            print(context_frame)
            
            # Combine forecasts
            prophet_forecast = context_frame['close_forecast']
            combined_forecast = combine_forecast(prophet_forecast, garch_forecast, data, mode=mode)
            print(combined_forecast)
            
            # Technical indicators
            data['rsi'] = ta.rsi(data['close'], 14)
            data['atr'] = ta.atr(data['high'], data['low'], data['close'], length=14)
            data['roc'] = ta.roc(data['close'], length=14)
            data['50ma'] = ta.ma('sma', data['close'], length=50)
            

            forecasted = combined_forecast['combined_forecast']
            first_close = forecasted.iloc[-3]
            last_close = forecasted.iloc[-1]
            
            print("Trend Direction and pips", last_close-first_close)
            print("close", round(last_close-first_close, 5))
            
            # Use findpeaks to detect peaks and troughs
            results = fp.fit(data.close)
            peaks = results['df'].loc[results['df']['peak'] == 1]
            valleys = results['df'].loc[results['df']['valley'] == 1]
            
            # Get the values of peaks and valleys
            highs = peaks['y'].values
            lows = valleys['y'].values
            print('Highs', highs)
            print("Lows", lows)
            
            # Check current market conditions
            current = data.iloc[-1]
            print(current)
            
            threshold = data['atr'].mean()  # Example threshold for volatility filter
            atr_value = current['atr']
            print('threshold ATR', round(threshold, 5))
            
            # # Convert ATR to pips
            # pip_conversion_factor = 0.0001 if symbol_name not in ['USDJPY', 'EURJPY', 'GBPJPY'] else 0.01
            # atr_in_pips = atr_threshold / pip_conversion_factor
            # print(atr_in_pips, 'atr in pips')
            
            # Define signals
            try:
                price = current['close']
                if  first_close > last_close and lows[-2] > highs[-1] and lows[-2] > lows[-1] and highs[-2] > highs[-1] and lows[-1] > highs[-1] and price < lows[-1]:   
                    print('Sell')
                    if current['rsi'] > 70 and atr_value < threshold:
                        SL = highs[-1] + (atr_value * 1.5)
                        TP = price - (abs(SL - highs[-1]) * 2.2)
                        lot_size = symbol_info['symbol min lot size']
                        
                        send_order(symbol = symbol_name, lot_size = lot_size, order_type = SELL, sl=SL, tp=TP)
                        
                elif first_close < last_close and lows[-2] < highs[-1] and lows[-2] < lows[-1] and highs[-2] < highs[-1] and lows[-1] < highs[-1] and price > highs[-1]:
                    print("Buy")
                    if current['rsi'] < 30  and atr_value < threshold:
                        SL = lows[-1] - (atr_value * 1.5)
                        TP = price + (abs(SL - lows[-1]) * 2.2)
                        lot_size = symbol_info['symbol min lot size']
                        send_order(symbol = symbol_name, lot_size = lot_size, order_type = BUY, sl=SL, tp=TP )
            except:
                pass
            
        else:  # mode == "test"
            data['rsi'] = ta.rsi(data['close'], 14)
            
            # ARIMA model fitting
            best_arima_order = pm.auto_arima(data['close'], seasonal=False, stepwise=True, trace=True).order
            arima_residual = arima_model_residual(data, best_arima_order)
            
            # GARCH model fitting
            best_garch_order = auto_garch_order(arima_residual)
            garch_forecast = garch_model_volatility(arima_residual, best_garch_order)
            
            # Prophet model training and forecasting
            context = train_models(data, mode='test')
            context_data = predict_trade(context)
            
            # Model evaluation
            evaluation_data = test_models(data, context_data)
            
            print(" - " * 25, symbol_name, " - " * 25)
            accuracy(evaluation_data)
            
            # Combine forecasts
            prophet_forecast = context_data['close_forecast']
            combined_forecast = combine_forecast(prophet_forecast, garch_forecast, data)
            
            # Prepare data for evaluation
            leng = abs(len(evaluation_data['actual_close']) - 30)
            actual_close_df = evaluation_data['actual_close'][leng:].reset_index(drop=True)
            checker = pd.concat([actual_close_df, combined_forecast], axis=1)
            checker.columns = ['actual', 'ml_predict']
            
            print(" - " * 25, symbol_name, " - " * 25)
            print(checker)
            
    end_time = time.perf_counter()
    print(f'Time Taken: {end_time - start_time} in sec')

def endBot():
    positions = mt5.positions_get()
    if positions == None:
        mt5.shutdown()
        quit()



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
    # startTask()
    start_bot()