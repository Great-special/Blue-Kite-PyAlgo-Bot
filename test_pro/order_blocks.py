import pandas as pd


class OrderBlock:
    def __init__(self, df, order_len) -> None:
        self.data = df
        self.order_len = order_len
    
    def steady(self):
        """
            Identify order blocks based on 4 consecutive candles where each candle couldn't close above or below the previous candle.
            
            Parameters:
                df (DataFrame): Market data containing 'Open', 'High', 'Low', and 'Close' columns.
            
            Returns:
                DataFrame: DataFrame containing the order blocks with max and min prices in the range.
        """
        order_blocks = []
        
        for i in range(len(self.data) - self.order_len):
            # Extract the 4 candles
            candle_1 = self.data.iloc[i]
            candle_2 = self.data.iloc[i + 1]
            candle_3 = self.data.iloc[i + 2]
            candle_4 = self.data.iloc[i + 3]
            
            # Check if the closing prices are within the previous candle's range
            if (
                candle_2['Close'] <= candle_1['High'] and candle_2['Close'] >= candle_1['Low'] and
                candle_3['Close'] <= candle_2['High'] and candle_3['Close'] >= candle_2['Low'] and
                candle_4['Close'] <= candle_3['High'] and candle_4['Close'] >= candle_3['Low']
            ):
                # Determine the direction
                direction = "Bullish" if candle_4['Close'] > candle_4['Open'] else "Bearish"
                # Get the max and min price within this range
                max_price = max(candle_1['High'], candle_2['High'], candle_3['High'], candle_4['High'])
                min_price = min(candle_1['Low'], candle_2['Low'], candle_3['Low'], candle_4['Low'])
                # Save the order block information
                order_blocks.append((self.data.index[i], self.data.index[i+3], direction, max_price, min_price))
        
        return pd.DataFrame(order_blocks, columns=['Start Date', 'End Date', 'Direction', 'Max Price', 'Min Price'])
        
        
        
        



import pandas as pd

class MarketStructureModel:
    def __init__(self, data, window_size=10):
        self.data = data
        self.window_size = window_size
        self.prev_high = None
        self.prev_low = None

    def identify_swing_points(self, window):
        current_high = window['High'].max()
        current_low = window['Low'].min()
        
        if self.prev_high is None or self.prev_low is None:
            # Initialize the first swing points
            self.prev_high = current_high
            self.prev_low = current_low
            return None, None, None, None

        # Check for HH, HL, LL, LH
        if current_high > self.prev_high:
            if current_low > self.prev_low:
                swing_type = 'HH-HL'  # Higher High, Higher Low
            else:
                swing_type = 'HH-LH'  # Higher High, Lower High
        elif current_low < self.prev_low:
            if current_high < self.prev_high:
                swing_type = 'LL-LH'  # Lower Low, Lower High
            else:
                swing_type = 'LL-HL'  # Lower Low, Higher Low
        else:
            swing_type = None
        
        # Update previous high/low for the next comparison
        self.prev_high = current_high
        self.prev_low = current_low

        return swing_type, current_high, current_low

    def identify_order_block(self, window):
        # Ensure the window is large enough
        if len(window) < 4:
            return None
        
        # Extract the relevant candles
        bearish_candle = window.iloc[-4]
        rejection_candle = window.iloc[-3]
        impulsive_candle = window.iloc[-2]
        breaking_candle = window.iloc[-1]
        
        # Check for a bullish order block
        if (bearish_candle['Close'] < bearish_candle['Open'] and
            rejection_candle['Close'] > rejection_candle['Open'] and
            rejection_candle['Low'] <= bearish_candle['Low'] and
            impulsive_candle['Close'] > impulsive_candle['Open'] and
            breaking_candle['Close'] > breaking_candle['Open']):
            
            # Calculate the body of the impulsive candle
            impulsive_body = impulsive_candle['Close'] - impulsive_candle['Open']
            
            # Check if breaking candle's low is higher than rejection candle's high by at least 50% of the impulsive body
            price_rejection_candle_high = rejection_candle['High']
            breaking_candle_low = breaking_candle['Low']
            if breaking_candle_low > price_rejection_candle_high + 0.5 * impulsive_body:
                order_block = {
                    'Order Block': 'Bullish',
                    'Order Block High': rejection_candle['High'],
                    'Order Block Low': rejection_candle['Low']
                }
                return order_block

        # Check for a bearish order block
        if (bearish_candle['Close'] > bearish_candle['Open'] and
            rejection_candle['Close'] < rejection_candle['Open'] and
            rejection_candle['High'] >= bearish_candle['High'] and
            impulsive_candle['Close'] < impulsive_candle['Open'] and
            breaking_candle['Close'] < breaking_candle['Open']):
            
            # Calculate the body of the impulsive candle
            impulsive_body = impulsive_candle['Open'] - impulsive_candle['Close']
            
            # Check if breaking candle's high is lower than rejection candle's low by at least 50% of the impulsive body
            price_rejection_candle_low = rejection_candle['Low']
            breaking_candle_high = breaking_candle['High']
            if breaking_candle_high < price_rejection_candle_low - 0.5 * impulsive_body:
                order_block = {
                    'Order Block': 'Bearish',
                    'Order Block High': rejection_candle['High'],
                    'Order Block Low': rejection_candle['Low']
                }
                return order_block

        return None

    def analyze_market(self):
        results = []
        for i in range(len(self.data) - self.window_size + 1):
            window = self.data.iloc[i:i + self.window_size]
            swing_type, current_high, current_low = self.identify_swing_points(window)
            order_block = self.identify_order_block(window)
            
            structure = {
                'Swing Type': swing_type,
                'Current High': current_high,
                'Current Low': current_low,
                'Order Block': order_block['Order Block'] if order_block else None,
                'Order Block High': order_block['Order Block High'] if order_block else None,
                'Order Block Low': order_block['Order Block Low'] if order_block else None,
            }
            
            structure['Start Index'] = i
            structure['End Index'] = i + self.window_size - 1
            results.append(structure)
        
        return pd.DataFrame(results)

# Example usage
data = pd.DataFrame({
    'Open': [1, 2, 3, 2, 1, 2, 3, 4, 3, 2, 5],
    'High': [2, 3, 4, 3, 2, 3, 4, 5, 4, 3, 6],
    'Low': [1, 1, 2, 1, 0, 1, 2, 3, 2, 1, 4],
    'Close': [2, 2, 3, 1, 1, 2, 4, 5, 3, 2, 6]
})

model = MarketStructureModel(data)
results = model.analyze_market()
print(results)

















# import pandas as pd

# def find_consolidation_order_blocks(df):
#     """
#     Function to find order blocks based on a sequence of 4 candles 
#     where each candle couldn't close above or below the previous candle.
    
#     Parameters:
#         df (DataFrame): Market data containing 'Open', 'High', 'Low', and 'Close' columns.
    
#     Returns:
#         DataFrame: DataFrame containing identified order blocks with the direction, max, and min prices.
#     """
#     order_blocks = []
    
#     for i in range(len(df) - 3):
#         # Check if the next 4 candles do not close above or below the previous candle
#         if (df['Close'][i+1] <= df['High'][i] and df['Close'][i+1] >= df['Low'][i]) and \
#            (df['Close'][i+2] <= df['High'][i+1] and df['Close'][i+2] >= df['Low'][i+1]) and \
#            (df['Close'][i+3] <= df['High'][i+2] and df['Close'][i+3] >= df['Low'][i+2]):
            
#             # Determine the direction based on the first and last candle in the sequence
#             direction = 'Bullish' if df['Close'][i+3] > df['Open'][i] else 'Bearish'
            
#             # Get the max and min price in this range
#             max_price = df['High'][i:i+4].max()
#             min_price = df['Low'][i:i+4].min()
            
#             order_blocks.append((df.index[i], direction, max_price, min_price))
    
#     return pd.DataFrame(order_blocks, columns=['Start Date', 'Direction', 'Max Price', 'Min Price'])

# # Sample data (replace this with your actual market data)
# data = {
#     'Open': [100, 105, 108, 110, 112, 109, 107, 108, 110, 111],
#     'High': [106, 109, 111, 114, 115, 110, 109, 111, 112, 113],
#     'Low': [99, 102, 105, 108, 110, 107, 104, 105, 108, 109],
#     'Close': [104, 107, 110, 112, 109, 108, 105, 109, 111, 110]
# }
# df = pd.DataFrame(data)

# # Call the function
# order_blocks = find_consolidation_order_blocks(df)

# # Display the identified order blocks
# print(order_blocks)



# import pandas as pd

# def find_order_blocks_4_candles(df):
#     """
#     Function to find order blocks based on a sequence of 4 candles where
#     each candle couldn't close above or below the previous candle.
    
#     Parameters:
#         df (DataFrame): Market data containing 'Open', 'High', 'Low', and 'Close' columns.
    
#     Returns:
#         DataFrame: DataFrame containing identified order blocks with the direction, max, and min prices.
#     """
#     order_blocks = []
    
#     for i in range(len(df) - 3):
#         # Check if the sequence of 4 candles meets the condition
#         if (
#             df['Close'][i] >= df['Close'][i+1] >= df['Close'][i+2] >= df['Close'][i+3] or
#             df['Close'][i] <= df['Close'][i+1] <= df['Close'][i+2] <= df['Close'][i+3]
#         ):
#             direction = "Bearish" if df['Close'][i] > df['Close'][i+3] else "Bullish"
#             max_price = df['High'][i:i+4].max()
#             min_price = df['Low'][i:i+4].min()
            
#             order_blocks.append({
#                 'Start Date': df.index[i],
#                 'End Date': df.index[i+3],
#                 'Direction': direction,
#                 'Max Price': max_price,
#                 'Min Price': min_price
#             })
    
#     return pd.DataFrame(order_blocks)

# # Sample data (replace this with your actual market data)
# data = {
#     'Open': [100, 102, 103, 104, 101, 100, 99, 98, 97],
#     'High': [105, 106, 107, 108, 102, 101, 100, 99, 98],
#     'Low': [99, 100, 101, 102, 99, 98, 97, 96, 95],
#     'Close': [102, 103, 104, 101, 100, 99, 98, 97, 96]
# }
# df = pd.DataFrame(data)

# # Call the function
# order_blocks = find_order_blocks_4_candles(df)

# # Display the identified order blocks
# print(order_blocks)
