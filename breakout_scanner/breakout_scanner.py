import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import numpy as np
import datetime as datetime
import matplotlib.pyplot as plt
from datetime import datetime


# NSE 500
nse500 = [
    "360ONE", "3MINDIA", "ABB", "ACC", "AIAENG", "APLAPOLLO", "AUBANK", "AARTIDRUGS", "AARTIIND", "AAVAS",
    "ABBOTINDIA", "ADANIENT", "ADANIGREEN", "ADANIPORTS", "ADANIPOWER", "ATGL", "AWL", "ABCAPITAL",
    "ABFRL", "AETHER", "AFFLE", "AJANTPHARM", "APLLTD", "ALKEM", "ALKYLAMINE", "AMBER",
    "AMBUJACEM", "ANGELONE", "ANURAS", "APARINDS", "APOLLOHOSP", "APOLLOTYRE", "APTUS", "ACI", "ASAHIINDIA", "ASHOKLEY",
    "ASIANPAINT", "ASTERDM", "ASTRAL", "ATUL", "AUROPHARMA", "AVANTIFEED", "DMART", "AXISBANK", "BASF", "BEML", "BLS",
    "BSE", "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BAJAJHLDNG", "BALAMINES", "BALKRISIND", "BALRAMCHIN", "BANDHANBNK",
    "BANKBARODA", "BANKINDIA", "MAHABANK", "BATAINDIA", "BAYERCROP", "BERGEPAINT", "BDL", "BEL", "BHARATFORG", "BHEL",
    "BPCL", "BHARTIARTL", "BIKAJI", "BIOCON", "BIRLACORPN", "BSOFT", "BLUEDART", "BLUESTARCO", "BBTC", "BORORENEW",
    "BOSCHLTD", "BRIGADE", "BCG", "BRITANNIA", "MAPMYINDIA", "CCL", "CESC", "CGPOWER", "CIEINDIA", "CRISIL", "CSBBANK",
    "CAMPUS", "CANFINHOME", "CANBK", "CGCL", "CARBORUNIV", "CASTROLIND", "CEATLTD", "CENTRALBK", "CDSL", "CENTURYPLY",
    "CENTURYTEX", "CERA", "CHALET", "CHAMBLFERT", "CHEMPLASTS", "CHOLAHLDNG", "CHOLAFIN", "CIPLA", "CUB", "CLEAN",
    "COALINDIA", "COCHINSHIP", "COFORGE", "COLPAL", "CAMS", "CONCOR", "COROMANDEL", "CRAFTSMAN", "CREDITACC", "CROMPTON",
    "CUMMINSIND", "CYIENT", "DCMSHRIRAM", "DLF", "DABUR", "DALBHARAT", "DATAPATTNS", "DEEPAKFERT", "DEEPAKNTR", "DELHIVERY",
    "DELTACORP", "DEVYANI", "DIVISLAB", "DIXON", "LALPATHLAB", "DRREDDY","EIDPARRY", "EIHOTEL", "EPL",
    "EASEMYTRIP", "EICHERMOT", "ELGIEQUIP", "EMAMILTD", "ENDURANCE", "ENGINERSIN", "EQUITASBNK", "ERIS", "ESCORTS",
    "EXIDEIND", "FDC", "NYKAA", "FEDERALBNK", "FACT", "FINEORG", "FINCABLES", "FINPIPE", "FSL", "FIVESTAR", "FORTIS",
    "GRINFRA", "GAIL", "GMMPFAUDLR", "GMRINFRA", "GALAXYSURF", "GARFIBRES", "GICRE", "GLAND", "GLAXO", "GLENMARK",
    "MEDANTA", "GOCOLORS", "GODFRYPHLP", "GODREJAGRO", "GODREJCP", "GODREJIND", "GODREJPROP", "GRANULES", "GRAPHITE",
    "GRASIM", "GESHIP", "GREENPANEL", "GRINDWELL", "GUJALKALI", "GAEL", "FLUOROCHEM", "GUJGASLTD", "GNFC", "GPPL", "GSFC",
    "GSPL", "HEG", "HCLTECH", "HDFCAMC", "HDFCBANK", "HDFCLIFE", "HFCL", "HLEGLAS", "HAPPSTMNDS", "HAVELLS", "HEROMOTOCO",
    "HIKAL", "HINDALCO", "HGS", "HAL", "HINDCOPPER", "HINDPETRO", "HINDUNILVR", "HINDZINC", "POWERINDIA", "HOMEFIRST",
    "HONAUT", "HUDCO", "ICICIBANK", "ICICIGI", "ICICIPRULI", "ISEC", "IDBI", "IDFCFIRSTB", "IDFC", "IFBIND", "IIFL", "IRB",
    "ITC", "ITI", "INDIACEM","IBREALEST", "INDIAMART", "INDIANB", "IEX", "INDHOTEL", "IOC", "IOB", "IRCTC",
    "IRFC", "INDIGOPNTS", "IGL", "INDUSTOWER", "INDUSINDBK", "INFIBEAM", "NAUKRI", "INFY", "INGERRAND", "INTELLECT",
    "INDIGO", "IPCALAB", "JBCHEPHARM", "JKCEMENT", "JBMA", "JKLAKSHMI", "JKPAPER", "JMFINANCIL", "JSWENERGY", "JSWSTEEL",
    "JAMNAAUTO", "JSL", "JINDALSTEL", "JINDWORLD", "JUBLFOOD", "JUBLINGREA", "JUBLPHARMA", "JUSTDIAL", "JYOTHYLAB",
    "KPRMILL", "KEI", "KNRCON", "KPITTECH", "KRBL", "KSB", "KAJARIACER", "KPIL", "KALYANKJIL", "KANSAINER", "KARURVYSYA",
    "KEC", "KENNAMET", "RUSTOMJEE", "KFINTECH", "KOTAKBANK", "KIMS","LTTS", "LICHSGFIN", "LTIM", "LAXMIMACH", "LT",
    "LATENTVIEW", "LAURUSLABS", "LXCHEM", "LEMONTREE", "LICI", "LINDEINDIA", "LUPIN", "LUXIND", "MMTC", "MRF",
    "MTARTECH", "LODHA", "MGL", "M&MFIN", "M&M", "MHRIL", "MAHLIFE", "MAHLOG", "MANAPPURAM", "MRPL", "MANKIND",
    "MARICO", "MARUTI", "MASTEK", "MFSL", "MAXHEALTH", "MAZDOCK", "MEDPLUS", "METROBRAND", "METROPOLIS",
    "MSUMI", "MOTILALOFS", "MPHASIS", "MCX", "MUTHOOTFIN", "NATCOPHARM", "NBCC", "NCC", "NHPC", "NLCINDIA", "NMDC",
    "NSLNISP", "NOCIL", "NTPC", "NH", "NATIONALUM", "NAVINFLUOR", "NAZARA", "NESTLEIND", "NETWORK18", "NAM-INDIA",
    "NUVOCO", "OBEROIRLTY", "ONGC", "OIL", "OLECTRA", "PAYTM", "OFSS", "ORIENTELEC", "POLICYBZR", "PCBL", "PIIND",
    "PNBHOUSING", "PNCINFRA", "PVRINOX", "PAGEIND", "PATANJALI", "PERSISTENT", "PETRONET", "PFIZER", "PHOENIXLTD",
    "PIDILITIND", "PEL", "PPLPHARMA", "POLYMED", "POLYCAB", "POLYPLEX", "POONAWALLA", "PFC", "POWERGRID", "PRAJIND",
    "PRESTIGE", "PRINCEPIPE", "PRSMJOHNSN", "PGHH", "PNB", "QUESS", "RBLBANK", "RECLTD", "RHIM", "RITES", "RADICO",
    "RVNL", "RAIN", "RAINBOW", "RAJESHEXPO", "RALLIS", "RCF", "RATNAMANI", "RTNINDIA", "RAYMOND", "REDINGTON", "RELAXO",
    "RELIANCE", "RBA", "ROSSARI", "ROUTE", "SBICARD", "SBILIFE", "SJVN", "SKFINDIA", "SRF", "MOTHERSON", "SANOFI",
    "SAPPHIRE", "SCHAEFFLER", "SHARDACROP", "SHOPERSTOP", "SHREECEM", "RENUKA", "SHRIRAMFIN", "SHYAMMETL", "SIEMENS",
    "SOBHA", "SOLARINDS", "SONACOMS", "SONATSOFTW", "STARHEALTH", "SBIN", "SAIL", "SWSOLAR", "STLTECH", "SUMICHEM",
    "SPARC", "SUNPHARMA", "SUNTV", "SUNDARMFIN", "SUNDRMFAST", "SUNTECK", "SUPRAJIT", "SUPREMEIND", "SUVENPHAR", "SUZLON",
    "SWANENERGY", "SYNGENE", "TCIEXP", "TTKPRESTIG", "TV18BRDCST", "TVSMOTOR", "TMB", "TANLA", "TATACHEM",
    "TATACOMM", "TCS", "TATACONSUM", "TATAELXSI", "TATAINVEST", "TATAMTRDVR", "TATAMOTORS", "TATAPOWER", "TATASTEEL",
    "TTML", "TEAMLEASE", "TECHM", "TEJASNET", "NIACL", "RAMCOCEM", "THERMAX", "TIMKEN", "TITAN", "TORNTPHARM", "TORNTPOWER",
    "TCI", "TRENT", "TRIDENT", "TRIVENI", "TRITURBINE", "TIINDIA", "UCOBANK", "UFLEX", "UNOMINDA", "UPL", "UTIAMC",
    "ULTRACEMCO", "UNIONBANK", "UBL","VGUARD", "VMART", "VIPIND", "VAIBHAVGBL", "VTL", "VARROC", "VBL",
    "MANYAVAR", "VEDL", "VIJAYA", "VINATIORGA", "IDEA", "VOLTAS", "WELCORP", "WESTLIFE", "WHIRLPOOL",
    "WIPRO", "YESBANK", "ZFCVINDIA", "ZEEL", "ZENSARTECH", "ZOMATO", "ZYDUSLIFE", "ZYDUSWELL", "ECLERX","MOIL"
]


start_date = "2020-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')  # You can use this for today's date

stock_data = {}

for stock in nse500:
    symbol = stock + ".NS"
    try:
        data = yf.download(symbol, start=start_date, end=end_date, interval="1wk")
        stock_data[stock] = data
    except Exception as e:
        print(f"Error downloading data for {stock}: {str(e)}")

for stock, data in stock_data.items():
    data.columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']






## Scan the level breakouts in last two weeks

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class StockAnalysis:
    def __init__(self, data):
        self.df = data.copy()
        self.df['Date'] = pd.to_datetime(self.df.index)  # Convert the index to datetime if it's not already
        self.df.set_index('Date', inplace=True)  # Set the 'Date' column as the index
        self.df = self.df[self.df['volume'] != 0]
        self.original_dates = self.df.index.strftime('%b %d, %y').tolist()
        self.df.reset_index(drop=True, inplace=True)

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

    def detect_zones(self, threshold=1, consolidation_threshold=0.5):
        """Detect support and resistance zones based on pivot points."""
        price_range = threshold * np.median(self.df['high'] - self.df['low'])
        pivot_points = self.df[self.df['isPivot'] > 0][['high', 'low']].values.flatten()
        pivot_points.sort()

        zones = []
        if pivot_points.size > 0:  # Check if pivot_points is not empty
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
        else:
            self.zones = []  # If no pivot points, then no zones

    def has_recent_breakout(self, days=15):
        """Check if there's a breakout in the last given number of days."""
        if not self.original_dates:  # Check if the list is empty
            return False

        recent_date = pd.to_datetime(self.original_dates[-1]) - pd.Timedelta(days=days)
        for breakout in self.breakouts:
            breakout_index, _ = breakout
            # Ensure that breakout_index is within the bounds of the original_dates list
            if breakout_index < len(self.original_dates):
                breakout_date = pd.to_datetime(self.original_dates[breakout_index])
                if breakout_date >= recent_date:
                    return True
        return False

    def plot_data(self, start=None, candles_back=500):
        end = len(self.df)
        start = max(end - candles_back, 0)
        dfpl = self.df[start:end]
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

recent_breakout_stocks = []

for stock, data in stock_data.items():
    data.columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    if data.empty:  # Skip if there's no data for the stock
        continue

    # Convert index to datetime if it isn't already
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    analysis = StockAnalysis(data)
    analysis.mark_pivots()
    analysis.detect_zones()
    analysis.detect_breakouts()
    if analysis.has_recent_breakout(days=15):
        recent_breakout_stocks.append(stock)

print("Stocks with a breakout in the last 2 weeks:")
print(recent_breakout_stocks)



##Plotting the breakout stocks

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Function to plot data for a single stock
def plot_stock(stock, data, candles_back=500):
    print(f"Plotting data for {stock}...")
    analysis = StockAnalysis(data)
    analysis.mark_pivots()
    analysis.detect_zones()
    analysis.detect_breakouts()
    analysis.plot_data(start=None, candles_back=candles_back)

# Loop through the breakout stocks and plot each one
for stock in recent_breakout_stocks:
    data = stock_data[stock]
    plot_stock(stock, data)



