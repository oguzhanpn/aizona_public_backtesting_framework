#%%
from datetime import datetime
from base_strategy import BaseStrategy
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from base_ml_model import BaseModel
import pandas as pd

class Model(BaseModel):
    """Model for the strategy"""
    def __init__(self):
        super().__init__()
        self.model = None
        self.feature_names = []
        
    def train(self, train_df):
        """Train the model using historical data
        
        Args:
            train_df: DataFrame with columns ['timestamp', 'price']
        """        
        # Need at least 180 data points for all features
        if len(train_df) < 180:
            raise ValueError("Need at least 180 data points for training")
        
        # Calculate features for the entire DataFrame
        features_df = self.prepare_features(train_df)
        
        # Calculate labels (price movement direction)
        # 1 for price increase, 0 for price decrease
        labels = (train_df['price'].shift(-1) > train_df['price']).astype(int)
        
        # Remove the last row since it won't have a label
        features_df = features_df.iloc[:-1]
        labels = labels.iloc[:-1]
        
        # Drop any rows with missing values
        valid_rows = features_df.notna().all(axis=1)
        X = features_df[valid_rows]
        y = labels[valid_rows]
        # Train the model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
        # Store feature names
        self.feature_names = features_df.columns.tolist()

    
    def _calculate_basic_features(self, df):
        """Calculate basic price-based features"""
        window_short = 69
        window_long = 180
        
        features = pd.DataFrame(index=df.index)
        features['sma_short'] = df['price'].rolling(window=window_short).mean()
        features['sma_long'] = df['price'].rolling(window=window_long).mean()
        features['price_change'] = df['price'].diff()
        features['price_change_pct'] = df['price'].pct_change()
        
        return features

    def _calculate_volatility_features(self, df, window=180):
        """Calculate volatility-based features"""
        features = pd.DataFrame(index=df.index)
        
        # Calculate rolling statistics
        rolling_mean = df['price'].rolling(window=window).mean()
        rolling_std = df['price'].rolling(window=window).std()
        
        features['volatility'] = rolling_std
        features['bollinger_upper'] = rolling_mean + (2 * rolling_std)
        features['bollinger_lower'] = rolling_mean - (2 * rolling_std)
        features['bollinger_bandwidth'] = (features['bollinger_upper'] - features['bollinger_lower']) / rolling_mean
        
        return features

    def _calculate_momentum_features(self, df):
        """Calculate momentum-based features"""
        features = pd.DataFrame(index=df.index)
        
        # RSI calculation
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=180).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=180).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['price'].rolling(window=90).mean()  # Simplified EMA
        ema_26 = df['price'].rolling(window=180).mean()
        features['macd'] = ema_12 - ema_26
        
        # Rate of Change (ROC)
        features['roc'] = df['price'].pct_change(periods=180) * 100
        
        return features

    def _calculate_trend_features(self, df):
        """Calculate trend-based features"""
        features = pd.DataFrame(index=df.index)
        
        # Rolling max and min
        rolling_high = df['price'].rolling(window=180).max()
        rolling_low = df['price'].rolling(window=180).min()
        
        # True Range (simplified)
        features['tr'] = rolling_high - rolling_low
        
        # Price position
        features['price_position'] = (df['price'] - rolling_low) / (rolling_high - rolling_low)
        
        return features

    def _calculate_time_features(self, df):
        """Calculate time-based features"""
        features = pd.DataFrame(index=df.index)
        
        # Convert timestamp to datetime
        timestamps = pd.to_datetime(df['timestamp'], unit='ms')
        
        features['hour'] = timestamps.dt.hour / 24.0
        features['day_of_week'] = timestamps.dt.dayofweek / 6.0
        features['day_of_month'] = timestamps.dt.day / 31.0
        
        return features

    def prepare_features(self, data_df):
        """Prepare features for the model using DataFrame operations
        
        Args:
            data_df: DataFrame with columns ['timestamp', 'ask_0_price', 'bid_0_price']
            
        Returns:
            DataFrame with calculated features
        """

        data_df['price'] = (data_df['ask_0_price'] + data_df['bid_0_price']) / 2

        # Calculate all feature sets
        feature_sets = [
            self._calculate_basic_features(data_df),
            self._calculate_volatility_features(data_df),
            self._calculate_momentum_features(data_df),
            self._calculate_trend_features(data_df),
            self._calculate_time_features(data_df)
        ]
        
        # Combine all feature DataFrames
        features = pd.concat(feature_sets, axis=1)
        
        return features


class Strategy(BaseStrategy):
    """Example ML model strategy Implementation"""
    
    strategy_name = "example_ml_strategy"
    configs = {
        # Strategy parameters
        "params": {
            'lookback_window': 180,
            'train_window': 18000,
            'pos_size': 1000,
        },
        
        # Backtest configuration
        "backtest_config": {
            "pair_list": ["BTC"],
            "start_date": datetime(2025, 2, 1),
            "end_date": datetime(2025, 2, 5),
            'resample_freq_in_ms': 1000,
            'use_ob_levels': 1,
            'make_ml_predictions': True
        }
    }

    def __init__(self, pair_list, params):
        super().__init__(pair_list, params)
        self.model_object = Model()

    def make_step_controls(self, pair):
        """Make step controls for the strategy"""
        if self.check_if_need_to_pass():
            return 
        
        if self.check_if_need_to_train():
            self.train_model()

        prediction = self.get_prediction(self.get_current_step(), 'example_ml_model')
        if prediction is None:
            #self.printer_method(f'Prediction is None. Skipping step: {self.get_current_step()}')
            return
        
        target_position_value = self.get_target_position_value(prediction)
        self.cancel_open_orders(pair)
        self.send_new_orders(pair, target_position_value)

    def send_new_orders(self, pair, target_position_value):
        """Execute trades for the strategy"""
        step = self.get_current_step()
        current_position = self.get_position(pair)
        current_price = self.get_mid_price(step)
        target_pos = target_position_value / current_price

        target_pos_diff = target_pos - current_position
        if target_pos_diff > 0:
            price = self.get_bid_price(step)
            _dict = {
                "pair": pair, 
                "side": "buy", 
                "price": price, 
                "amount": abs(target_pos_diff),
                "step": step
            }
            self.create_limit_order(_dict)
        elif target_pos_diff < 0:
            price = self.get_ask_price(step)
            _dict = {
                "pair": pair, 
                "side": "sell", 
                "price": price, 
                "amount": abs(target_pos_diff),
                "step": step
            }
            self.create_limit_order(_dict)
    
    def get_target_position_value(self, prediction):
        """Get the target position value for the strategy"""
        if prediction == 1:
            return self.params['pos_size']
        else:
            return -self.params['pos_size']

    def train_model(self):
        """Train the model"""
        historical_data = self.get_historical_data(
            step = self.get_current_step(),
            lookback_in_seconds=self.params['train_window'],
            pair = self.pair_list[0]
            )
        self.model_object.train(historical_data)
        self.printer_method(f"Model trained with {len(historical_data)} data points")

        self.make_future_predictions('example_ml_model', self.model_object)

    def check_if_need_to_pass(self):
        """Check if the strategy needs to pass"""
        return self.get_current_step() < self.params['train_window']
    
    def check_if_need_to_train(self):
        """Check if the strategy needs to train"""
        return self.model_object.model is None


if __name__ == "__main__":
    from runing_tools import run_strategy_on_server

    username = "test_user" #Your username here
    password = "" #Your password here
    strategy_file_name = "example_ml_strategy.py"

    run_strategy_on_server(
        username=username,
        password=password,
        script_path=strategy_file_name,
        file_name=strategy_file_name
    )