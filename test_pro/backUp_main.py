import numpy as np
import itertools
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
print("-+-"*20)
# get data on MetaTrader 5 version
# print(mt5.version())
# print("-+-"*20)
# print(mt5.account_info()._asdict())

D1 = mt5.TIMEFRAME_D1
H1 = mt5.TIMEFRAME_H1
H4 = mt5.TIMEFRAME_H4
M30 = mt5.TIMEFRAME_M30
M15 = mt5.TIMEFRAME_M15
M5 = mt5.TIMEFRAME_M5
BUY = mt5.ORDER_TYPE_BUY
BUY_LIMIT = mt5.ORDER_TYPE_BUY_LIMIT
SELL = mt5.ORDER_TYPE_SELL
SELL_LIMIT = mt5.ORDER_TYPE_SELL_LIMIT

TRADEABLE = ['XAUUSD', 'EURUSD', 'GBPJPY', 'NZDCHF', 'APPLE', 'GOOGLE', 'COCA-COLA', 'AMAZON', 'Platinum', 'BTCUSD']

print('-'*25)


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
    rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
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
    print(arima_forecast)
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
    df_high = prophet_df[['time', 'high']].rename(columns={'time':'ds', 'high':'y'})
    df_low = prophet_df[['time', 'low']].rename(columns={'time':'ds', 'low':'y'})
    
    
    model_close = Prophet()
    model_close.fit(df_close)
    model_open = Prophet()
    model_open.fit(df_open)
    model_low = Prophet()
    model_low.fit(df_low)
    model_high = Prophet()
    model_high.fit(df_high)
    
    future_close = model_close.make_future_dataframe(period, freq="60Min", include_history=False)
    future_open = model_open.make_future_dataframe(period, freq="60Min", include_history=False)
    future_low = model_low.make_future_dataframe(period, freq="60Min", include_history=False)
    future_high = model_high.make_future_dataframe(period, freq='60Min', include_history=False)
    
    context = {
        'model_close':model_close,
        'future_close':future_close,
        'model_open':model_open,
        'future_open':future_open,
        'model_low':model_low,
        'future_low':future_low,
        'model_high':model_high,
        'future_high':future_high
    }
    return context


def train_models(data, mode="live"):
    if mode != "live":
        train_len = int(len(data)*0.7)
        train = data[:train_len]
        period = abs(len(data) - train_len)
        return analyze_data(train, period)
    else:
        return analyze_data(data)
        


def predict_trade(context):
    model_close = context['model_close']
    model_open = context['model_open']
    model_high = context['model_high']
    model_low = context['model_low']
    future_close = context['future_close']
    future_open = context['future_open']
    future_high = context['future_high']
    future_low = context['future_low']
    
    close_forecast = model_close.predict(future_close)
    open_forecast = model_open.predict(future_open)
    high_forecast = model_high.predict(future_high)
    low_forecast = model_low.predict(future_low)
    context_data = {
        'time':open_forecast['ds'],
        'open_forecast':open_forecast['yhat'],
        'high_forecast':high_forecast['yhat'],
        'low_forecast':low_forecast['yhat'],
        'close_forecast':close_forecast['yhat'],
    }
    return context_data
    

def test_models(data, context):
    test_len = int(len(data)*0.7)
    test = data[test_len:]
    
    test = test.reset_index(drop=True)
    context_data = context
    open_forecast = context_data['open_forecast']
    high_forecast = context_data['high_forecast']
    low_forecast = context_data['low_forecast']
    close_forecast = context_data['close_forecast']
    
    evaluation = pd.DataFrame(
        {
            'actual_date':test['time'],
            
            'actual_open':test['open'],
            'predicted_open':open_forecast,
            '%Diff_open':((open_forecast - test['open'])/test['open']) * 100,
            
            'actual_high':test['high'],
            'predicted_high':high_forecast,
            '%Diff_high':((high_forecast - test['high'])/test['high']) * 100,
            
            'actual_low':test['low'],
            'predicted_low':low_forecast,
            '%Diff_low':((low_forecast - test['low'])/test['low']) * 100,
            
            'actual_close':test['close'],
            'predicted_close':close_forecast,
            '%Diff_close':((close_forecast - test['close'])/test['close']) * 100,
            
            'direction': 'Up' if (open_forecast  <close_forecast).all() else "Down"
            
        }
    )
    print(evaluation.head(30))
    print(evaluation.tail(30))
    return evaluation


def combine_forecast(prophet_forecast, garch_forecast, data, price='close', horizon=30):
    # Check columns of garch_forecast
    # print("Columns in garch_forecast:")
    # print(garch_forecast.variance.tail())
    # print(garch_forecast.mean.columns)
    # print(garch_forecast.mean.tail())
    print("Shape of prophet_forecast.values:", prophet_forecast.values.shape)
    print("Shape of garch_forecast.mean:", garch_forecast.mean['h.01'].shape)
    
    # Expand garch_forecast_mean to match the shape of prophet_forecast_values
    garch_forecast_mean_expanded = np.repeat(garch_forecast.mean['h.01'], prophet_forecast.values.shape[0])

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
    print(combined_forecast_ml)
    # Evaluation
    actual = y_test
    mae_ml = mean_absolute_error(actual, combined_forecast_ml)
    mse_ml = mean_squared_error(actual, combined_forecast_ml)
    rmse_ml = np.sqrt(mse_ml)
    mape_ml = mean_absolute_percentage_error(actual, combined_forecast_ml)
    accuracy = 100 * (1 -mape_ml)

    print(f'MAE (ML): {mae_ml}, MSE (ML): {mse_ml}, RMSE (ML): {rmse_ml}')
    print(f'Accuracy (ML): {accuracy:.2f}')
    
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


def send_order(symbol, lot_size, open_price, close_price):
    lot = lot_size
    point = mt5.symbol_info(symbol).point
    price = mt5.symbol_info_tick(symbol).ask if mt5.symbol_info_tick(symbol).ask < open_price else open_price
    print(point)
    deviation = 20
    pips = int(abs(open_price-close_price))
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "sl": price - (pips * point),
        "tp": price + (pips * point),
        "deviation": deviation,
        "magic": 234000,
        "comment": "python script open",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }

    # send a trading request
    result = mt5.order_send(request)
    # check the execution result
    print("1. order_send(): by {} {} lots at {} with deviation={} points".format(symbol,lot,price,deviation));
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print("2. order_send failed, retcode={}".format(result.retcode))
        # request the result as a dictionary and display it element by element
        result_dict=result._asdict()
        for field in result_dict.keys():
            print("   {}={}".format(field,result_dict[field]))
            # if this is a trading request structure, display it element by element as well
            if field=="request":
                traderequest_dict=result_dict[field]._asdict()
                for tradereq_filed in traderequest_dict:
                    print("       traderequest: {}={}".format(tradereq_filed,traderequest_dict[tradereq_filed]))
        print("shutdown() and quit")
        mt5.shutdown()
        quit()
    else:
        print("2. order_send done, ", result)
        print("   opened position with POSITION_TICKET={}".format(result.order))
        print("   sleep 2 seconds before closing position #{}".format(result.order))
    
    
    

def start_bot():
    # mode can be live or test
    mode = 'test'
    # data = pd.read_excel("D:\my projects\Light Wave\Blue Kite\BTCUSDH1.xlsx")
    symbol_name = TRADEABLE[1]
    print(" - "*25, symbol_name, " - "*20)
    res = get_symbol_data(symbol_name, 5040)
    data = res['rates_frame']
    print(data)
    if not mode == "test":
        modified_data = data.copy()
        context = train_models(data)
        context_data = predict_trade(context)
        context_frame = pd.DataFrame(context_data)
        context_frame['Trend'] = "Up" if (context_frame['open_forecast'] < context_frame['close_forecast']).all() else "Down"
        print(context_frame)
        modified_data['rsi'] = ta.rsi(modified_data['close'], 14)
        print(modified_data.tail(20))
        
    else:
        data['rsi'] = ta.rsi(data['close'], 14)
        
        best_arima_order = pm.auto_arima(data['close'], seasonal=False, stepwise=True, trace=True).order
        arima_residual  = arima_model_residual(data, best_arima_order)
        best_grach_order = auto_garch_order(arima_residual)
        garch_forecast = garch_model_volatility(arima_residual, best_grach_order)
        
        context = train_models(data, mode='test')
        context_data = predict_trade(context)
    
        evaluation_data = test_models(data, context_data)
        
        print(" - "*25, symbol_name, " - "*25)
        accuracy(evaluation_data)
        
        prophet_forecast = context_data['close_forecast']
        combined_forecast = combine_forecast(prophet_forecast, garch_forecast, data)
        # train_len = int(len(data)*0.7)
        # train = data[:train_len]
        # period = abs(len(data) - train_len)
        leng = abs(len(evaluation_data['actual_close']) - 30)
        
        # combined_forecast_df = pd.DataFrame(combined_forecast, columns=['ml_predict'])

        # Resetting the index of actual_close series
        actual_close_df = evaluation_data['actual_close'][leng:].reset_index(drop=True)

        # Combining the DataFrames
        checker = pd.concat([actual_close_df, combined_forecast], axis=1)

        # Renaming the columns if necessary
        checker.columns = ['actual', 'ml_predict']
        print(" - "*25, symbol_name, " - "*25)
        print(checker)        
        


def end_bot():
    pass


start_bot()