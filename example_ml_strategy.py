#%%
from datetime import datetime
from base_strategy import BaseStrategy
import numpy as np
from xgboost import XGBClassifier
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

        # Random downsample the data
        features_df, labels = self.random_downsample(features_df, labels)
        
        # Remove the last row since it won't have a label
        features_df = features_df.iloc[:-1]
        labels = labels.iloc[:-1]
        
        # Drop any rows with missing values
        valid_rows = features_df.notna().all(axis=1)
        X = features_df[valid_rows]
        y = labels[valid_rows]
        # Train the model
        self.model = XGBClassifier(n_estimators=100, random_state=56)
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

    def get_performance_metrics(self, data_df):
        """Get performance metrics for the model"""
        predictions = pd.Series(self.predict(data_df), index=data_df.index)
        actual = (data_df['price'].shift(-1) > data_df['price']).astype(int)
        results = pd.DataFrame({'predictions': predictions, 'actual': actual}).dropna()
        
        # Calculate metrics using proper boolean operations
        accuracy = np.mean(results['predictions'] == results['actual'])
        true_positives = np.sum((results['predictions'] == 1) & (results['actual'] == 1))
        predicted_positives = np.sum(results['predictions'] == 1)
        actual_positives = np.sum(results['actual'] == 1)
        
        precision = true_positives / predicted_positives if predicted_positives > 0 else 0
        recall = true_positives / actual_positives if actual_positives > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {    
            'accuracy': np.round(accuracy, 2),
            'precision': np.round(precision, 2),
            'recall': np.round(recall, 2),
            'f1_score': np.round(f1_score, 2)
        }

    def random_downsample(self, features_df, labels):
        """Randomly downsample the data"""
        # Randomly select balanced data
        pos_labels = labels[labels == 1]    
        neg_labels = labels[labels == 0]
        sample_size = min(len(pos_labels), len(neg_labels))
        pos_indices = np.random.choice(range(len(pos_labels)), size=sample_size, replace=False)
        neg_indices = np.random.choice(range(len(neg_labels)), size=sample_size, replace=False)
        selected_indices = np.concatenate([pos_indices, neg_indices])
        return features_df.iloc[selected_indices], labels.iloc[selected_indices]

class Strategy(BaseStrategy):
    """Example ML model strategy Implementation"""
    
    strategy_name = "example_ml_strategy"
    configs = {
        # Strategy parameters
        "params": {
            'lookback_window': 180,
            'train_window': 18000,
            'pos_size': 1000,
            "take_profit_pct_threshold": 0.001
        },
        
        # Backtest configuration
        "backtest_config": {
            "pair_list": ["BTC"],
            "start_date": datetime(2024, 1, 1),
            "end_date": datetime(2024, 1, 2),
            'resample_freq_in_ms': 1000,
            'use_ob_levels': 1,
            'make_ml_predictions': True
        }
    }

    def __init__(self, pair_list, params):
        super().__init__(pair_list, params)
        self.model_object = Model()
        self.is_profit_taken = True
        self.last_take_profit_pos = 0

    def make_step_controls(self, pair):
        """Make step controls for the strategy"""
        if self.check_if_need_to_pass():
            return 
        
        if self.check_if_need_to_train():
            self.train_model()

        prediction = self.get_prediction(self.get_current_step(), 'example_ml_model')
        if prediction is None:
            # self.printer_method(f'Prediction is None. Skipping step: {self.get_current_step()}')
            return
        
        target_position_value = self.get_target_position_value(prediction)
        self.cancel_open_orders(pair)
        self.send_new_orders(pair, target_position_value)

        if self.get_current_step() % 100000 == 0:
            self.send_accuracy_metrics()

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
                "step": step,
                "maker": True
            }
            self.create_limit_order(_dict)
        elif target_pos_diff < 0:
            price = self.get_ask_price(step)
            _dict = {
                "pair": pair, 
                "side": "sell", 
                "price": price, 
                "amount": abs(target_pos_diff),
                "step": step,
                "maker": True
            }
            self.create_limit_order(_dict)
    
    def get_target_position_value(self, prediction):
        """Get the target position value for the strategy"""
        current_pos = self.get_position(self.pair_list[0])
        current_profit_pct = self.get_pos_profit_pct(self.pair_list[0])
        
        if current_pos == 0:
            self.is_profit_taken = True

        if self.is_profit_taken:
            if self.last_take_profit_pos < 0: 
                if prediction == 0:
                    # Already taken profit, no need to open new position
                    return 0
                else:
                    # side change, open new position
                    self.is_profit_taken = False
                    self.last_take_profit_pos = 0
                    return self.params['pos_size']
            else:
                if prediction == 0:
                    # side change, open new position
                    self.is_profit_taken = False
                    self.last_take_profit_pos = 0
                    return self.params['pos_size']
                else:
                    # Already taken profit, no need to open new position
                    return 0
        else:
            if current_profit_pct > self.params['take_profit_pct_threshold']:
                self.is_profit_taken = True
                self.last_take_profit_pos = current_pos
                return 0
            else:
                return current_pos

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

    def send_accuracy_metrics(self):
        """Send accuracy metrics to the server"""
        historical_data = self.get_historical_data(
            step = self.get_current_step(),
            lookback_in_seconds=self.params['train_window'],
            pair = self.pair_list[0]
        )
        performance_metrics = self.model_object.get_performance_metrics(historical_data)
        current_ts_in_ms = self.get_data('timestamp', self.get_current_step())
        current_datetime = datetime.fromtimestamp(current_ts_in_ms / 1000)
        self.printer_method(f'{current_datetime} last 24 hour model performance metrics:\n {performance_metrics}')

    def get_pos_profit_pct(self, pair):
        """Get the profit of the position"""
        current_position = self.get_position(pair)
        current_price = self.get_mid_price(self.get_current_step())
        pos_entry_price = self.get_open_position_price(pair)

        if current_position == 0:
            return 0

        pos_entry_val = current_position * pos_entry_price
        current_pos_val = current_position * current_price
        profit = current_pos_val - pos_entry_val
        return profit / abs(pos_entry_val)

    
if __name__ == "__main__":
    from runing_tools import run_strategy_on_server

    username = "oguzhan" #Your username here
    password = "i|6:Au$}E!3w" #Your password here
    strategy_file_name = "example_ml_strategy.py"

    run_strategy_on_server(
        username=username,
        password=password,
        script_path=strategy_file_name,
        file_name=strategy_file_name
    )