#%%
from datetime import datetime
from base_strategy import BaseStrategy
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from base_ml_model import BaseModel

class Model(BaseModel):
    """Model for the strategy"""
    def __init__(self):
        super().__init__()
        self.model = None
        self.feature_names = []

    def train(self, train_data):
        """Train the model using historical data
        
        Args:
            train_data: List of tuples (timestamp, price)
        """
        # Need at least 180 data points for all features
        if len(train_data) < 180:
            raise ValueError("Need at least 180 data points for training")
        
        # Prepare features and labels for each data point
        features_list = []
        labels = []
        
        # Start from index 180 to have enough lookback data
        for i in range(180, len(train_data)):
            # Get historical data window for feature calculation
            historical_window = train_data[i-180:i+1]
            
            # Calculate features for this window
            features = self.prepare_features(historical_window)
            
            # Calculate label (price movement direction)
            # 1 for price increase, 0 for price decrease
            if i < len(train_data) - 1:  # Ensure we have next price for label
                current_price = train_data[i][1]
                next_price = train_data[i+1][1]
                label = 1 if next_price > current_price else 0
                
                # Only add to training data if we have all features
                if len(features) > 0:
                    features_list.append(list(features.values()))
                    labels.append(label)
        
        # Convert to numpy arrays for training
        X = np.array(features_list)
        y = np.array(labels)
        
        # Train the model (example using RandomForestClassifier)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
        # Store feature names for future predictions
        self.feature_names = list(features.keys())

    def predict(self, data):
        """Predict the next step"""
        # Prepare features
        features = self.prepare_features(data)

        # Predict the next step
        features_array = np.array(list(features.values())).reshape(1, -1)
        if len(features) != len(self.feature_names):
            print('there are missing features. \n Expected: ', self.feature_names, '\n Got: ', features)
            return None
        
        _pred = self.model.predict(features_array)
        return _pred[0]
    
    def _calculate_basic_features(self, prices):
        """Calculate basic price-based features"""
        features = {}
        
        # Moving averages
        window_short = 69
        window_long = 180
        features['sma_short'] = sum(prices[-window_short:]) / window_short
        features['sma_long'] = sum(prices[-window_long:]) / window_long
        
        # Price momentum
        features['price_change'] = prices[-1] - prices[-2]
        features['price_change_pct'] = (prices[-1] - prices[-2]) / prices[-2]
        
        return features

    def _calculate_volatility_features(self, prices, window=180):
        """Calculate volatility-based features"""
        features = {}
        
        if len(prices) >= window:
            mean_price = sum(prices[-window:]) / window
            features['volatility'] = (
                sum((p - mean_price) ** 2 for p in prices[-window:]) 
                / window
            ) ** 0.5
            
            # Add Bollinger Bands
            std_dev = features['volatility']
            features['bollinger_upper'] = mean_price + (2 * std_dev)
            features['bollinger_lower'] = mean_price - (2 * std_dev)
            features['bollinger_bandwidth'] = (features['bollinger_upper'] - features['bollinger_lower']) / mean_price
        
        return features

    def _calculate_momentum_features(self, prices):
        """Calculate momentum-based features"""
        features = {}
        
        if len(prices) >= 180:
            # RSI calculation
            changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            gains = [c if c > 0 else 0 for c in changes]
            losses = [-c if c < 0 else 0 for c in changes]
            avg_gain = sum(gains[-180:]) / 180
            avg_loss = sum(losses[-180:]) / 180
            rs = avg_gain / avg_loss if avg_loss != 0 else 0
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            if len(prices) >= 180:
                ema_12 = sum(prices[-90:]) / 90  # Simplified EMA
                ema_26 = sum(prices[-180:]) / 180
                features['macd'] = ema_12 - ema_26
                
            # Rate of Change (ROC)
            if len(prices) >= 180:
                features['roc'] = ((prices[-1] - prices[-180]) / prices[-180]) * 100
        
        return features

    def _calculate_trend_features(self, prices):
        """Calculate trend-based features"""
        features = {}
        
        if len(prices) >= 180:
            # Average Directional Index (ADX) - Simplified version
            highs = prices[:-1]  # Using close prices as proxy for highs
            lows = prices[:-1]   # Using close prices as proxy for lows
            
            # True Range (TR)
            tr = max(max(highs) - min(lows), abs(max(highs) - prices[-1]), abs(min(lows) - prices[-1]))
            features['tr'] = tr
            
            # Price position relative to recent range
            highest_high = max(prices[-180:])
            lowest_low = min(prices[-180:])
            features['price_position'] = (prices[-1] - lowest_low) / (highest_high - lowest_low)
        
        return features

    def _calculate_time_features(self, data):
        """Calculate time-based features"""
        features = {}
        # Extract timestamp from the first element of each tuple
        timestamp = datetime.fromtimestamp(data[-1][0]/1000)
        
        features['hour'] = timestamp.hour / 24.0  # Normalize to [0,1]
        features['day_of_week'] = timestamp.weekday() / 6.0  # Normalize to [0,1]
        features['day_of_month'] = timestamp.day / 31.0  # Normalize to [0,1]
        
        return features

    def prepare_features(self, data):
        """Prepare features for the model
        
        Args:
            data: List of tuples (timestamp, price)
            
        Returns:
            Dictionary of features
        """
        prices = [price_data[1] for price_data in data]
        
        # Initialize features dictionary
        features = {}
        
        # Collect features from all calculation methods
        feature_sets = [
            self._calculate_basic_features(prices),
            self._calculate_volatility_features(prices),
            self._calculate_momentum_features(prices),
            self._calculate_trend_features(prices),
            self._calculate_time_features(data)
        ]
        
        # Combine all feature dictionaries
        for feature_set in feature_sets:
            features.update(feature_set)
        
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
            'use_ob_levels': 1
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
        
        data = self.get_historical_data(
            data_type = 'ask_0_price',
            step = self.get_current_step(),
            lookback_in_seconds=self.params['lookback_window']+30, # Get 30 seconds more data for potential missing data points
            pair = pair
            )
        prediction = self.model_object.predict(data)
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
            data_type = 'ask_0_price',
            step = self.get_current_step(),
            lookback_in_seconds=self.params['train_window'],
            pair = self.pair_list[0]
            )
        self.model_object.train(historical_data)
        self.printer_method(f"Model trained with {len(historical_data)} data points")

    
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