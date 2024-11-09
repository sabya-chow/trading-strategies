import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import linregress
from datetime import datetime
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# NSE 500
symbol = "3MINDIA.NS"
import yfinance as yf
from datetime import datetime
start_date = "2015-01-06"
end_date = datetime.now().strftime('%Y-%m-%d')  # You can use this for today's date
data = yf.download(symbol, start=start_date, end=end_date, interval="1wk")
data.columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']




class StockAnalysis:
    def __init__(self, data):
        self.df = data
        self.original_dates = data.index.strftime('%b %d, %y').tolist()
        #self.df = self.df[self.df['volume'] != 0]
        self.df.reset_index(drop=True, inplace=True)
        self.zones = []  # To store detected support and resistance zones
        self.levels = []  # To store detected price levels

    def consolidate_zones(self, zones, consolidation_range):
        """Consolidate nearby zones into larger ones based on the consolidation range."""
        consolidated_zones = []
        current_zone = zones[0]

        for next_zone in zones[1:]:
            # Check if the next zone is close enough to be consolidated with the current one
            if next_zone[0] - current_zone[1] <= consolidation_range:
                # Extend the current zone to include the next one
                current_zone = (current_zone[0], max(current_zone[1], next_zone[1]))
            else:
                # Add the current zone to the list and move to the next one
                consolidated_zones.append(current_zone)
                current_zone = next_zone

        # Add the last zone to the list
        consolidated_zones.append(current_zone)
        return consolidated_zones

    def detect_zones(self, threshold=1, consolidation_threshold=0.5):
        """Detect support and resistance zones based on pivot points."""
        price_range = threshold * np.median(self.df['high'] - self.df['low'])
        pivot_points = self.df[self.df['isPivot'] > 0][['high', 'low']].values.flatten()
        pivot_points.sort()

        zones = []
        current_zone = [pivot_points[0]]

        for point in pivot_points[1:]:
            if point - current_zone[-1] <= price_range:
                current_zone.append(point)
            else:
                zones.append((min(current_zone), max(current_zone)))
                current_zone = [point]

        if current_zone:
            zones.append((min(current_zone), max(current_zone)))

        # Consolidate zones
        consolidated_zones = self.consolidate_zones(zones, consolidation_threshold * np.median(
            self.df['high'] - self.df['low']))

        # The zone dictionary should include 'start', 'end', and 'value'
        self.zones = [{'start': start, 'end': end, 'value': (start, end)} for start, end in consolidated_zones]

    def isPivot(self, candle, window=10):
        if candle - window < 0 or candle + window >= len(self.df):
            return 0

        pivotHigh, pivotLow = 1, 2
        for i in range(candle - window, candle + window + 1):
            if self.df.iloc[candle].low > self.df.iloc[i].low:
                pivotLow = 0
            if self.df.iloc[candle].high < self.df.iloc[i].high:
                pivotHigh = 0

        if pivotHigh and pivotLow:
            return 3
        elif pivotHigh:
            return 1
        elif pivotLow:
            return 2
        else:
            return 0

    def detect_breakouts(self, volume_factor=1.5):
        """Detect breakout candles through the resistance zones with significant volume."""
        breakouts = []

        # Iterate through each resistance zone
        for zone in self.zones:
            zone_start = zone['start']
            zone_end = zone['end']

            # Flag to indicate if the last candle was within or below the resistance zone
            within_or_below_zone = False

            # Filter the dataframe to find candles that are potentially breaking out
            for index, row in self.df.iterrows():
                # Check if the candle is within or below the resistance zone
                if row['high'] <= zone_end:
                    within_or_below_zone = True
                # If it is above, we check if the previous was within or below and if it's green and if volume is high enough
                elif row['close'] > row['open'] and within_or_below_zone and row['close'] > zone_end:
                    if row['volume'] >= volume_factor * np.median(self.df['volume']):
                        # Check if this is a valid breakout
                        if self.confirm_breakout(index):
                            breakouts.append((index, row['close']))
                            # Reset the flag as we have found a breakout
                            within_or_below_zone = False

        self.breakouts = breakouts

    def confirm_breakout(self, breakout_candle, continuation_candles=1):
        """Confirm the breakout with continuation of price movement."""
        breakout_low = self.df.iloc[breakout_candle]['low']
        following_candles = self.df[breakout_candle + 1: breakout_candle + continuation_candles + 1]

        if not following_candles.empty and all(
                breakout_low < candle['low'] or candle['close'] >= breakout_low for _, candle in
                following_candles.iterrows()):
            return True
        return False

    def mark_pivots(self, window=10):
        self.df['isPivot'] = self.df.apply(lambda x: self.isPivot(x.name, window), axis=1)
        self.df['pointpos'] = self.df.apply(lambda row: self.pointpos(row), axis=1)

    def pointpos(self, x):
        if x['isPivot'] == 2:
            return x['low'] - 1e-3
        elif x['isPivot'] == 1:
            return x['high'] + 1e-3
        else:
            return np.nan

    def plot_data(self, start=None, end=None, candles_back=None):
        if end is None:
            end = len(self.df)  # Set end to the last index of the dataframe
        if start is None:
            if candles_back is not None:
                start = max(end - candles_back, 0)  # Look 'candles_back' from the end
            else:
                start = 0  # Or start from the first index
        dfpl = self.df.iloc[start:end].copy()
        dfpl_dates = self.original_dates[start:end]  # Match the length of dfpl

        # Use the stored original dates for plotting
        dfpl.index = self.original_dates[start:end]

        # Create a subplot for the candlestick and the volume
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.03, subplot_titles=('Price', 'Volume'),
                            row_width=[0.2, 0.7])

        # Add Candlestick plot
        fig.add_trace(go.Candlestick(
            x=dfpl.index,
            open=dfpl['open'],
            high=dfpl['high'],
            low=dfpl['low'],
            close=dfpl['close'],
            name="Candlestick"), row=1, col=1)

        # Add Volume bar plot
        fig.add_trace(go.Bar(
            x=dfpl.index,
            y=dfpl['volume'],
            name="Volume"), row=2, col=1)

        # Add pivot points as scatter plot
        fig.add_trace(go.Scatter(
            x=dfpl.index,
            y=dfpl['pointpos'],
            mode="markers",
            marker=dict(size=5, color="yellow"),
            name="pivot"), row=1, col=1)

        # Hide the rangeslider for a cleaner look
        fig.update_layout(xaxis_rangeslider_visible=False, showlegend=False)

        # Loop through the zones and add them as shapes
        for zone in self.zones:
            fig.add_shape(
                type="rect",
                x0=dfpl.index[start],
                x1=dfpl.index[end-1],
                y0=zone['start'],
                y1=zone['end'],
                fillcolor='yellow',
                opacity=0.5,
                line=dict(color='blue'),
                row=1, col=1
            )

        # Plot breakout markers
        for breakout in self.breakouts:
            breakout_index, breakout_price = breakout
            if start <= breakout_index < end:
                fig.add_trace(go.Scatter(
                    x=[dfpl.index[breakout_index]],
                    y=[breakout_price],
                    mode="markers",
                    marker=dict(size=10, color="green"),
                    name="breakout"), row=1, col=1)

        # Set x-axis to show dates
        fig.update_xaxes(type='category')

        # Update layout to make the plot wider
        fig.update_layout(width=1000, height=600)

        fig.show()


# Example of how to use the class:
# Ensure you have 'data' which is a DataFrame with the necessary structure
analysis = StockAnalysis(data)
analysis.mark_pivots()
analysis.detect_zones()  # (threshold=0.05)  # Set threshold as desired
analysis.detect_breakouts()
analysis.plot_data()

