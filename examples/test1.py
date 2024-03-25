import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime
from finrl import config
from finrl import config_tickers
import os
from finrl.main import check_and_make_directories
from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
)

#%matplotlib inline
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from finrl.meta.data_processor import DataProcessor
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from pprint import pprint
import sys
sys.path.append("../FinRL")
import itertools

CACHE_DIR = "cache"


def fetch_prices(market_name, ticker_list, start_date_str, end_date_str):
    try:
        file_name = f"{market_name}_{start_date_str}_{end_date_str}.csv"
        path = os.path.join(CACHE_DIR, file_name)
        if os.path.exists(path):
            prices_df = pd.read_csv(path)
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
    # Data Preprocessing
    feature_engineer = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=True,
        use_turbulence=True,
        user_defined_feature=False)

    processed_df = feature_engineer.preprocess_data(prices_df)
    return processed_df


################################################################
# MAIN

# Create directories
check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])


# Set training start- and end dates
TRAIN_START_DATE = '2010-01-01'
TRAIN_END_DATE = '2021-10-01'
TRADE_START_DATE = '2021-10-01'
TRADE_END_DATE = '2024-03-01'

# Download Dow Jones price data using Yahoo Finance
market_name = "NASDAQ100"
ticker_list = config_tickers.NAS_100_TICKER

# Fetch prices
prices_df = fetch_prices(market_name, ticker_list, TRAIN_START_DATE, TRADE_END_DATE)
print(f"Fetched len(prices_df) records")

# Perform data preprocessing
processed_priced_df = perform_preprocessing(prices_df)
processed_priced_df.sort_values(['date', 'tic'], ignore_index=True).head(10)




