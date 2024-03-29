#from __future__ import annotations
import os
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.meta.paper_trading.alpaca import PaperTradingAlpaca
from finrl.meta.paper_trading.common import train, test, alpaca_history, DIA_history
from finrl.config import INDICATORS
from finrl.config_tickers import NAS_100_TICKER  # Import NASDAQ 100 tickers
from finrl.config_tickers import DOW_30_TICKER
import datetime
from pandas.tseries.offsets import BDay, DateOffset
import matplotlib.pyplot as plt

# Set Alpaca base URLs
ALPACA_DATA_API_BASE_URL = "https://data.alpaca.markets/v2"
ALPACA_DATA_API_KEY = os.environ['ALPACA_TRADING_API_KEY']
ALPACA_DATA_API_SECRET = os.environ['ALPACA_TRADING_API_SECRET']

ALPACA_TRADING_API_BASE_URL = "https://paper-api.alpaca.markets"  # Live trading: https://api.alpaca.markets
ALPACA_TRADING_API_KEY = os.environ['ALPACA_TRADING_API_KEY']
ALPACA_TRADING_API_SECRET = os.environ['ALPACA_TRADING_API_SECRET']


def get_alpaca_api_keys_from_env():
    # Fetches the Alpaca API keys from environment variables
    alpaca_api_keys_map = {}
    return alpaca_api_keys_map


def calculate_start_end_dates():
    today = datetime.datetime.today()

    test_end_date = (today - BDay(1)).to_pydatetime().date()
    test_start_date = (test_end_date - BDay(1)).to_pydatetime().date()
    train_end_date = (test_start_date - BDay(1)).to_pydatetime().date()
    train_start_date = (train_end_date - BDay(5)).to_pydatetime().date()
    train_full_start_date = train_start_date
    train_full_end_date = test_end_date

    train_start_date_str = str(train_start_date)
    train_end_date_str = str(train_end_date)
    test_start_date_str = str(test_start_date)
    test_end_date_str = str(test_end_date)
    train_full_start_date_str = str(train_full_start_date)
    train_full_end_date_str = str(train_full_end_date)
    return train_start_date_str, train_end_date_str, test_start_date_str, test_end_date_str, train_full_start_date_str, train_full_end_date_str


def configure_indicators(indicators):
    indicator_list = indicators
    # Add deltas
    indicator_list.append('close_-1_d')
    indicator_list.append('close_-2_d')
    indicator_list.append('close_-3_d')
    indicator_list.append('close_-4_d')

    # Lags
    indicator_list.append('close_-2_s')

    # Add Bollinger bands with different window sizes
    indicator_list.append('boll_20')
    indicator_list.append('boll_3')
    indicator_list.append('boll_5')
    indicator_list.append('boll_7')

    return indicator_list


def plot_performance(returns_erl, returns_dia):
    plt.figure(dpi=1000)
    plt.grid()
    plt.grid(which="minor", axis="y")
    plt.title("Stock Trading (Paper trading)", fontsize=20)
    plt.plot(returns_erl, label="ElegantRL Agent", color="red")
    # plt.plot(returns_sb3, label = 'Stable-Baselines3 Agent', color = 'blue' )
    # plt.plot(returns_rllib, label = 'RLlib Agent', color = 'green')
    plt.plot(returns_dia, label="DJIA", color="grey")
    plt.ylabel("Return", fontsize=16)
    plt.xlabel("Year 2021", fontsize=16)
    plt.xticks(size=14)
    plt.yticks(size=14)
    """
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(78))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(6))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.005))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=2))
    ax.xaxis.set_major_formatter(
        ticker.FixedFormatter(["", "10-19", "", "10-20", "", "10-21", "", "10-22"])
    )
    """
    plt.legend(fontsize=10.5)
    plt.tight_layout()
    plt.savefig("papertrading_stock.png")
    plt.close()

################################################################
# MAIN
################################################################

# Initialize environment
env = StockTradingEnv

# ERL Configuration
# if you want to use larger datasets (change to longer period), and it raises error, please try to increase "target_step". It should be larger than the episode steps.
ERL_PARAMS = {
    "learning_rate": 3e-6,
    "batch_size": 2048,
    "gamma": 0.985,
    "seed": 312,
    "net_dimension": [128, 64],
    "target_step": 5000,
    "eval_gap": 30,
    "eval_times": 1,
}

# Calculate dates
train_start_date_str, \
train_end_date_str, \
test_start_date_str, \
test_end_date_str, \
train_full_start_date_str, \
train_full_end_date_str = calculate_start_end_dates()

# Configure indicators
indicator_list = configure_indicators(INDICATORS)
#indicator_list = INDICATORS

# Define ticker list
#ticker_list = ['AMZN', 'AAPL', 'TSLA', 'MSFT'] #  NAS_100_TICKER
ticker_list = DOW_30_TICKER

#  Configuration
data_source = "alpaca"
model_name = "ppo"
time_interval = "1Min"
drl_lib = "elegantrl"
cwd = "./papertrading_erl"  # Current working directory
break_step = 1e5
use_vix = True
turbulence_thresh = 30
max_stock = 1e2


# Training
train(
    start_date=train_start_date_str,
    end_date=train_end_date_str,
    ticker_list=ticker_list,
    data_source=data_source,
    time_interval=time_interval,
    technical_indicator_list=indicator_list,
    drl_lib=drl_lib,
    env=env,
    model_name=model_name,
    if_vix=use_vix,
    API_KEY=ALPACA_DATA_API_KEY,
    API_SECRET=ALPACA_DATA_API_SECRET,
    API_BASE_URL=ALPACA_DATA_API_BASE_URL,
    erl_params=ERL_PARAMS,
    cwd=cwd,  # current_working_dir
    break_step=break_step,
)

# Use the trained agent with test data
account_value_erl = test(
    start_date=test_start_date_str,
    end_date=test_end_date_str,
    ticker_list=ticker_list,
    data_source=data_source,
    time_interval=time_interval,
    technical_indicator_list=indicator_list,
    drl_lib=drl_lib,
    env=env,
    model_name=model_name,
    if_vix=True,
    API_KEY=ALPACA_DATA_API_KEY,
    API_SECRET=ALPACA_DATA_API_SECRET,
    API_BASE_URL=ALPACA_DATA_API_BASE_URL,
    cwd=cwd,
    net_dimension=ERL_PARAMS["net_dimension"],
)

#  Train agent on all data
train(
    start_date=train_full_start_date_str,
    end_date=train_full_end_date_str,
    ticker_list=ticker_list,
    data_source=data_source,
    time_interval=time_interval,
    technical_indicator_list=indicator_list,
    drl_lib=drl_lib,
    env=env,
    model_name=model_name,
    if_vix=True,
    API_KEY=ALPACA_DATA_API_KEY,
    API_SECRET=ALPACA_DATA_API_SECRET,
    API_BASE_URL=ALPACA_DATA_API_BASE_URL,
    erl_params=ERL_PARAMS,
    cwd=cwd,
    break_step=break_step,
)

action_dim = len(ticker_list)
state_dim = (
    1 + 2 + 3 * action_dim + len(INDICATORS) * action_dim
)  # Calculate the DRL state dimension manually for paper trading. amount + (turbulence, turbulence_bool) + (price, shares, cd (holding time)) * stock_dim + tech_dim


paper_trading_erl = PaperTradingAlpaca(
    ticker_list=DOW_30_TICKER,
    time_interval=time_interval,
    drl_lib=drl_lib,
    agent=model_name,
    cwd=cwd,
    net_dim=ERL_PARAMS["net_dimension"],
    state_dim=state_dim,
    action_dim=action_dim,
    API_KEY=ALPACA_TRADING_API_KEY,
    API_SECRET=ALPACA_TRADING_API_SECRET,
    API_BASE_URL=ALPACA_TRADING_API_BASE_URL,
    tech_indicator_list=indicator_list,
    turbulence_thresh=turbulence_thresh,
    max_stock=max_stock,
)

paper_trading_erl.run()

"""
# Check Portfolio Performance
df_erl, cumu_erl = alpaca_history(
    key=ALPACA_DATA_API_KEY,
    secret=ALPACA_DATA_API_SECRET,
    url=ALPACA_DATA_API_BASE_URL,
    start="2022-09-01",  # must be within 1 month
    end="2022-09-12",
)  # change the date if error occurs

df_djia, cumu_djia = DIA_history(start="2022-09-01")
returns_erl = cumu_erl - 1
returns_dia = cumu_djia - 1
returns_dia = returns_dia[: returns_erl.shape[0]]


# Plot performance
plot_performance(returns_erl, returns_dia)
"""




print("All done!")
