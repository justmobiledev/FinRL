import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from finrl import config_tickers
import os
from finrl.main import check_and_make_directories
from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS
)

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from finrl.plot import backtest_stats, get_baseline
from pypfopt.efficient_frontier import EfficientFrontier
import sys
sys.path.append("../FinRL")


def fetch_prices(market_name, ticker_list, start_date_str, end_date_str):
    try:
        file_name = f"{market_name}_{start_date_str}_{end_date_str}.csv"
        path = os.path.join(DATA_SAVE_DIR, file_name)
        if os.path.exists(path):
            prices_df = pd.read_csv(path)
            #  Remove unnamed columns
            prices_df = prices_df.loc[:, ~prices_df.columns.str.contains('^Unnamed')]
            return prices_df
        else:
            #  Fetch remotely
            prices_df = YahooDownloader(start_date=start_date_str,
                                 end_date=end_date_str,
                                 ticker_list=ticker_list).fetch_data()

            #  Cache file
            prices_df.to_csv(path)

            return prices_df
    except Exception as e:
        print(f"Failed to fetch price data for {market_name}, error: {str(e)}")
        return None


def perform_preprocessing(prices_df):
    print(f"Indicators used: {INDICATORS}")
    feature_engineer = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=True,
        use_turbulence=True,
        user_defined_feature=False)

    processed_df = feature_engineer.preprocess_data(prices_df)
    return processed_df


def perform_train_test_split(prices_df, train_start_date, train_end_date, trade_start_date, trade_end_date):
    train_df = data_split(prices_df, train_start_date, train_end_date)
    trade_df = data_split(prices_df, trade_start_date, trade_end_date)
    train_length = len(train_df)
    trade_length = len(trade_df)
    print(f"Training dataframe length: {train_length}")
    print(f"Trading dataframe length: {trade_length}")
    return train_df, trade_df


def train_a2c_agent(env_train):
    agent = DRLAgent(env=env_train)

    model_a2c = agent.get_model("a2c")

    # Set up logger
    tmp_path = RESULTS_DIR + '/a2c'
    new_logger_a2c = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model_a2c.set_logger(new_logger_a2c)

    # Train agent
    total_timesteps = 3000
    trained_a2c = agent.train_model(model=model_a2c,
                                     tb_log_name='a2c',
                                     total_timesteps=total_timesteps)
    return trained_a2c


def process_df_for_mvo(df):
    df = df.sort_values(['date', 'tic'], ignore_index=True)[['date', 'tic', 'close']]
    fst = df
    fst = fst.iloc[0:stock_dimension, :]
    tic = fst['tic'].tolist()

    mvo = pd.DataFrame()

    for k in range(len(tic)):
        mvo[tic[k]] = 0

    for i in range(df.shape[0] // stock_dimension):
        n = df
        n = n.iloc[i * stock_dimension:(i + 1) * stock_dimension, :]
        date = n['date'][i * stock_dimension]
        mvo.loc[date] = n['close'].tolist()

    return mvo


# Codes in this section partially refer to Dr G A Vijayalakshmi Pai
# https://www.kaggle.com/code/vijipai/lesson-5-mean-variance-optimization-of-portfolios/notebook
def compute_stock_returns(stock_prices, rows, columns):
    stock_returns = np.zeros([rows - 1, columns])
    for j in range(columns):  # j: Assets
        for i in range(rows - 1):  # i: Daily Prices
            stock_returns[i, j] = ((stock_prices[i + 1, j] - stock_prices[i, j]) / stock_prices[i, j]) * 100
    return stock_returns


def calculate_asset_returns(stock_data_df):
    # Calculate stock returns
    ar_stock_prices = np.asarray(stock_data_df)
    [rows, cols] = ar_stock_prices.shape
    ar_returns = compute_stock_returns(ar_stock_prices, rows, cols)

    # compute mean returns and variance covariance matrix of returns
    mean_returns = np.mean(ar_returns, axis=0)
    covariance_returns = np.cov(ar_returns, rowvar=False)
    return mean_returns, covariance_returns


def plot_results(result_df):
    plt.rcParams["figure.figsize"] = (15, 5)
    plt.figure()
    plt.tight_layout()
    result_df.plot()

    # Store output
    path = os.path.join(RESULTS_DIR, "results_plot.png")
    plt.savefig(path)
    plt.close()


################################################################
# MAIN
################################################################
# Create directories
check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])

# Set training start- and end dates - last 10 years
TRAIN_START_DATE = '2014-01-01'
TRAIN_END_DATE = '2021-10-01'
TRADE_START_DATE = '2021-10-01'
TRADE_END_DATE = '2024-03-01'

# Download NASDAQ 100 price data from Yahoo Finance
market_name = "NASDAQ100"
ticker_list = config_tickers.NAS_100_TICKER

# Fetch prices
prices_df = fetch_prices(market_name, ticker_list, TRAIN_START_DATE, TRADE_END_DATE)
print(f"Fetched len(prices_df) records")

# Perform data preprocessing
processed_priced_df = perform_preprocessing(prices_df)
processed_priced_df.sort_values(['date', 'tic'], ignore_index=True).head(10)

# Train-test split
train_df, trade_df = perform_train_test_split(processed_priced_df,
                                              TRAIN_START_DATE,
                                              TRAIN_END_DATE,
                                              TRADE_START_DATE,
                                              TRADE_END_DATE)

# Determine feature dimensions
stock_dimension = len(train_df.tic.unique())
state_space = 1 + (2 * stock_dimension + len(INDICATORS) * stock_dimension)
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

# Create configuration for training
initial_balance = 100000
commission = 0.0  # percent
buy_cost_list = sell_cost_list = [commission] * stock_dimension
num_stock_shares = [0] * stock_dimension
hmax = 100
env_kwargs = {
    "hmax": hmax,
    "initial_amount": initial_balance,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}

# Initialize training environment
e_train_gym = StockTradingEnv(df=train_df, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))

# Train A2C agent
trained_a2c = train_a2c_agent(env_train)

# Initialize trading environment
e_trade_gym = StockTradingEnv(df=trade_df, turbulence_threshold=70, risk_indicator_col='vix', **env_kwargs)

# Make predictions
trained_model_a2c = trained_a2c
df_account_value_a2c, df_actions_a2c = DRLAgent.DRL_prediction(model=trained_model_a2c, environment=e_trade_gym)

# Prepare dataframes for Mean Variance optimization
train_mvo_df = data_split(processed_priced_df, TRAIN_START_DATE, TRAIN_END_DATE).reset_index()
trade_mvo_df = data_split(processed_priced_df, TRADE_START_DATE, TRADE_END_DATE).reset_index()

stock_data_df = process_df_for_mvo(train_mvo_df)
trade_data_df = process_df_for_mvo(trade_mvo_df)

# Calculate mean, covariance returns
mean_returns, covariance_returns = calculate_asset_returns(stock_data_df)

ef_mean = EfficientFrontier(mean_returns, covariance_returns, weight_bounds=(0, 0.5))
raw_weights_mean = ef_mean.max_sharpe()
cleaned_weights_mean = ef_mean.clean_weights()
mvo_weights = np.array([1000000 * cleaned_weights_mean[i] for i in range(stock_data_df.shape[1])])

# Create an array of the last portfolio prices
last_price = np.array([1/p for p in stock_data_df.tail(1).to_numpy()[0]])

# Multiply last price by MVO weights
initial_portfolio = np.multiply(mvo_weights, last_price)

portfolio_assets = trade_data_df @ initial_portfolio
mvo_result_df = pd.DataFrame(portfolio_assets, columns=["Mean Var"])

# Store MVO results
path = os.path.join(RESULTS_DIR, "mvo_result_df.csv")
mvo_result_df.to_csv(path)

# Get baseline stats using market index
market_baseline_df = get_baseline(
        ticker="^NDX",
        start=TRADE_START_DATE,
        end=TRADE_END_DATE)
market_baseline_stats = backtest_stats(market_baseline_df, value_col_name='close')
print("Market basline stats:")
print(market_baseline_stats)

# Prepare the A2C prediction results for merging
df_result_a2c = df_account_value_a2c.set_index(df_account_value_a2c.columns[0])
df_result_a2c.rename(columns={'account_value': 'A2C account value'}, inplace=True)

# Calculate account value
market_df = pd.DataFrame()
market_df['date'] = df_account_value_a2c['date']
market_df['account_value'] = market_baseline_df['close'] / market_baseline_df['close'][0] * env_kwargs["initial_amount"]
market_df.rename(columns={'account_value': 'market account value'}, inplace=True)
market_df.set_index('date', inplace=True)

# Merge market results with A2C results
result_df = pd.DataFrame()
result_df = pd.merge(result_df, df_result_a2c, how='outer', left_index=True, right_index=True)
result_df = pd.merge(result_df, market_df, how='outer', left_index=True, right_index=True)

# Plot results
plot_results(result_df)

print("All done!")
