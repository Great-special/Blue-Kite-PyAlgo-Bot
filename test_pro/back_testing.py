import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
from findpeaks import findpeaks
import matplotlib.pyplot as plt
from main_prophet import (
    arima_model_residual, pm, auto_garch_order,
    garch_model_volatility, train_models, combine_forecast, predict_trade
)


class BlueKite(Strategy):
    upper_bound = 70
    lower_bound = 30
    period = 30
    ta_window = 14
    ma_window = 50
    lookahead = 14
    atr_multiplier = 1.5
    tp_multiplier = 2.2
    mode = 'live'
    position_size = 1

    def init(self):
        # Ensure data is in pandas DataFrame format
        self.data_df = pd.DataFrame({
            'time': self.data.index,
            'open': self.data.Open,
            'close': self.data.Close,
            'high': self.data.High,
            'low': self.data.Low
        })

        # Calculate indicators using pandas_ta
        rsi_values = ta.rsi(self.data_df['close'], length=self.ta_window)
        atr_values = ta.atr(self.data_df['high'], self.data_df['low'], self.data_df['close'], length=self.ta_window)
        roc_values = ta.roc(self.data_df['close'], length=self.ta_window)
        ma_values = ta.roc(self.data_df['close'], length=self.ma_window)

        # Ensure indicators return valid arrays
        if rsi_values is None or atr_values is None or roc_values is None or ma_values is None:
            raise ValueError("One of the indicators returned None")

        # Ensure indicators have the correct length
        if len(rsi_values) != len(self.data_df) or len(atr_values) != len(self.data_df) or len(roc_values) != len(self.data_df) or len(roc_values) != len(self.data_df):
            raise ValueError("Indicator length does not match data length")

        # Check if any of the indicators have the wrong dimension
        if rsi_values.ndim != 1:
            raise ValueError("RSI values have incorrect dimension")
        if atr_values.ndim != 1:
            raise ValueError("ATR values have incorrect dimension")
        if roc_values.ndim != 1:
            raise ValueError("ROC values have incorrect dimension")
        if ma_values.ndim != 1:
            raise ValueError("MA values have incorrect dimension")

        # Convert pandas Series to numpy array and use self.I correctly
        self.rsi = self.I(lambda: rsi_values.to_numpy())
        self.atr = self.I(lambda: atr_values.to_numpy())
        self.roc = self.I(lambda: roc_values.to_numpy())
        self.ma = self.I(lambda: ma_values.to_numpy())
        self.threshold = atr_values.mean()
        print("RSI, ATR, ROC indicators initialized.")

        # Get ARIMA model residuals
        self.best_arima_order = pm.auto_arima(self.data_df['close'], seasonal=False, stepwise=True, trace=True).order
        self.arima_residual = arima_model_residual(self.data_df, self.best_arima_order, price='close')
        print(f"Best ARIMA order: {self.best_arima_order}")
        print(f"ARIMA residuals calculated.")

        # Get GARCH model forecast
        self.best_garch_order = auto_garch_order(self.arima_residual)
        self.garch_forecast = garch_model_volatility(self.arima_residual, self.best_garch_order)
        print(f"Best GARCH order: {self.best_garch_order}")
        print(f"GARCH forecast calculated.")

        # # Initialize findpeaks
        # self.fp = findpeaks(method='topology', lookahead=self.lookahead, whitelist=['peak', 'valley'])
        # print("Findpeaks initialized.")
        


    def next(self):
        self.on_candle_data_df = pd.DataFrame({
            'time': self.data.index,
            'open': self.data.Open,
            'close': self.data.Close,
            'high': self.data.High,
            'low': self.data.Low
        })
        size = int(len(self.on_candle_data_df) * 0.3)
        # Prophet model training and forecasting
        context = train_models(self.on_candle_data_df.iloc[size:], self.mode)
        context_data = predict_trade(context)
        context_frame = pd.DataFrame(context_data)
        context_frame['Trend'] = context_frame['open_forecast'] < context_frame['close_forecast']
        context_frame['Trend'] = context_frame['Trend'].map({True: "Up", False: "Down"})
        print("Prophet model training and forecasting completed.")

        # Combine forecasts
        prophet_forecast = context_frame['close_forecast']
        combined_forecast, slope = combine_forecast(prophet_forecast, self.garch_forecast, self.data_df, mode=self.mode)

        self.forecasted = combined_forecast['combined_forecast']
        self.first_close = self.forecasted.iloc[-3]
        self.last_close = self.forecasted.iloc[-1]
        print(len(self.forecasted))
        print(f"First and last close forecasts extracted: {self.first_close}, {self.last_close}")

        self.buy_count_pk_va = 0
        self.sell_count_pk_va = 0
        
        self.price = self.data.Close[-1]
        self.atr_threshold = self.atr[-1]
        self.low = self.data.Low[-1]
        self.high = self.data.High[-1]
        print(self.price, self.atr_threshold, self.high, self.low)
        # # Use findpeaks to detect peaks and troughs
        # results = self.fp.fit(self.data_df.close)
        # self.peaks = results['df'].loc[results['df']['peak'] == 1]
        # self.valleys = results['df'].loc[results['df']['valley'] == 1]

        
        # # Get the values of peaks and valleys
        # highs = self.peaks['y'].values
        # lows = self.valleys['y'].values

        # Print the values of peaks and valleys
        # print("Peaks:", highs)
        # print("Valleys:", lows)

       
        # self.fp.plot()
        # try:
        #     if lows[-2] > highs[-1] and lows[-2] > lows[-1] and highs[-2] > highs[-1] and lows[-1] > highs[-1] and self.price < lows[-1]:       
        #         if self.rsi[-1] > self.upper_bound and self.atr_threshold < self.threshold:
        #             self.SL = highs[-1] + (self.atr[-1] * self.atr_multiplier)
        #             self.TP = self.price - (abs(self.SL - highs[-1]) * self.tp_multiplier)
        #             self.sell(sl=self.SL, tp=self.TP)
        #             # print(f"Sell conditions: SL={self.SL}, TP={self.TP}, RSI={self.rsi[-1]}, Price={self.price}, ATR Threshold={self.atr_threshold}")
        #             # print("Sell order executed.")

        #     if lows[-2] < highs[-1] and lows[-2] < lows[-1] and highs[-2] < highs[-1] and lows[-1] < highs[-1] and self.price > highs[-1]:
        #         if self.rsi[-1] < self.lower_bound and self.atr_threshold < self.threshold:
        #             self.SL = lows[-1] - (self.atr[-1] * self.atr_multiplier)
        #             self.TP = self.price + (abs(self.SL - lows[-1]) * self.tp_multiplier)
        #             self.buy(sl=self.SL, tp=self.TP)
        #             # print(f"Buy conditions: SL={self.SL}, TP={self.TP}, RSI={self.rsi[-1]}, Price={self.price}, ATR Threshold={self.atr_threshold}")
        #             # print("Buy order executed.")
        # except:
        #     pass
        
        if self.last_close < self.price and self.rsi[-1] < self.lower_bound: # slope > 0 last_close >  MA  first < last -TP SL(price-last)/2.2 + atr
            self.SL = self.low - (self.atr_threshold * 2)
            self.TP = self.high + (1.8 * abs(self.price - self.SL))
            print(self.SL, self.TP, 'Buy')
            self.buy(sl=self.SL, tp=self.TP)
        
        if self.last_close > self.price and self.rsi[-1] > self.upper_bound: # slope < 0  last_close < MA  first > last - TP SL(price-last)/2.2 + atr
            self.SL = self.high + (self.atr_threshold * 2)
            self.TP = self.low - (1.8 * abs(self.price - self.SL))
            print(self.SL, self.TP, 'sell')
            self.sell(sl=self.SL, tp=self.TP)
        
# Load your data into a pandas DataFrame
df = pd.read_excel("D:\my projects\Light Wave\Blue Kite\EURUSDH1.xlsx")
print(df.columns)
print(df.head())
data = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]

data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'})
data = data.set_index('time')
print(data.head())

# Run the backtest
bt = Backtest(data, BlueKite, cash=100000)

# Optimize the parameters
# stats = bt.optimize(
#     # upper_bound=range(60, 80, 4),
#     # lower_bound=range(20, 40, 4),
#     # ta_window=range(10, 20, 4),
#     lookahead=range(5, 15, 2),
#     # atr_multiplier=[1.0, 1.5, 2.0],
#     # tp_multiplier=[1.5, 2.0, 2.2, 2.5],
#     # constraint=lambda param: param.upper_bound > param.lower_bound,
#     maximize='Equity Final [$]'
# )
# print(stats)

# Access the best parameters
# best_params = stats._strategy.lookahead
# print(stats._strategy)
# print("Best parameters: n1 =", best_params)

stats = bt.run()
print(stats)
# bt.plot()