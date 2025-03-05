import ccxt as xt

# instantiation of ccxt
exchange = xt.binance()
print(exchange.features)
exchange.set_sandbox_mode(True)