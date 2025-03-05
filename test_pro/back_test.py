import pandas as pd
from ta.momentum import RSIIndicator, ROCIndicator
from ta.volatility import AverageTrueRange
from backtesting import Backtest, Strategy
from backtesting.test import EURUSD
from main_prophet import (arima_model_residual, pm, auto_garch_order, garch_model_volatility, train_models, predict_trade)

class BlueKite(Strategy):
    upper_bound = 70
    lower_bound = 30
    ta_window = 14
    mode = 'live'
    position_size = 0.1
    
    def init(self):
        # Ensure data is in pandas DataFrame format
        self.data_df = pd.DataFrame({
            'time': self.data.index,
            'open': self.data.Open,
            'close': self.data.Close,
            'high': self.data.High,
            'low': self.data.Low
        })

        # Calculate indicators using ta library
        rsi_indicator = RSIIndicator(close=self.data_df['close'], window=self.ta_window)
        atr_indicator = AverageTrueRange(high=self.data_df['high'], low=self.data_df['low'], close=self.data_df['close'], window=self.ta_window)
        roc_indicator = ROCIndicator(close=self.data_df['close'], window=self.ta_window)

        self.rsi = self.I(rsi_indicator.rsi)
        self.atr = self.I(atr_indicator.average_true_range)
        self.roc = self.I(roc_indicator.roc)
        self.threshold = self.atr.mean()

        # Get ARIMA model residuals
        self.best_arima_order = pm.auto_arima(self.data_df['close'], seasonal=False, stepwise=True, trace=True).order
        self.arima_residual = arima_model_residual(self.data_df, self.best_arima_order, price='close')

        # Get GARCH model forecast
        self.best_garch_order = auto_garch_order(self.arima_residual)
        self.garch_forecast = garch_model_volatility(self.arima_residual, self.best_garch_order)

        # Prophet model training and forecasting
        context = train_models(self.data_df, self.mode)
        context_data = predict_trade(context)
        context_frame = pd.DataFrame(context_data)
        context_frame['Trend'] = context_frame['open_forecast'] < context_frame['close_forecast']
        context_frame['Trend'] = context_frame['Trend'].map({True: "Up", False: "Down"})

        # Extract forecasts
        open_forecast = context_frame['open_forecast']
        close_forecast = context_frame['close_forecast']

        self.first_open = open_forecast.iloc[0]
        self.last_open = open_forecast.iloc[-1]
        self.first_close = close_forecast.iloc[0]
        self.last_close = close_forecast.iloc[-1]

    def next(self):
        self.price = self.data.Close[-1]
        self.atr_threshold = self.atr[-1]
        self.pips = abs(self.first_open - self.last_close)

        if self.first_open > self.last_open and self.first_close > self.last_close:
            print('Sell')
            self.SL = self.price + (self.pips + (self.atr[-1] * 2))
            self.TP = self.price - (self.pips + (self.atr[-1] * 2) * 1.8)
            if self.rsi[-1] > self.upper_bound and self.roc[-1] < 0 and self.atr_threshold < self.threshold:
                lot_size = self.position_size
                self.sell(size=lot_size, sl=self.SL, tp=self.TP)
        elif self.first_open < self.last_open and self.first_close < self.last_close:
            print("Buy")
            self.SL = self.price - (self.pips + (self.atr[-1] * 2))
            self.TP = self.price + (self.pips + (self.atr[-1] * 2) * 1.8)
            if self.rsi[-1] < self.lower_bound and self.roc[-1] > 0 and self.atr_threshold < self.threshold:
                lot_size = self.position_size
                self.buy(size=lot_size, sl=self.SL, tp=self.TP)

# Load your data into a pandas DataFrame
data = pd.DataFrame(EURUSD)

# Run the backtest
bt = Backtest(data, BlueKite, cash=10000, commission=0.002)
stats = bt.run()
print(stats)
