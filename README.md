# Strategy Development Guide for Aizona Trading Framework

## Overview
The Aizona trading framework provides a structured way to develop trading strategies using a base class (`BaseStrategy`) that handles common trading operations. New strategies should inherit from this base class and implement specific trading logic.

## Basic Structure

```python
class MyStrategy(BaseStrategy):
    def __init__(self, pair_list, params):
        super().__init__(pair_list, params)

    def make_step_controls(self, pair):
        # Implement your strategy logic here
        pass

```


## Available Methods from BaseStrategy

### Position Management
- `get_position(pair)`: Get current position size for a trading pair
- `get_open_position_price(pair)`: Get the entry price of the current position

### Order Management
- `create_limit_order(order_dict)`: Create a new limit order
- `cancel_open_orders(pair)`: Cancel all open orders for a pair
- `get_buy_orders(pair)`: Get all open buy orders
- `get_sell_orders(pair)`: Get all open sell orders
- `get_buy_order_count(pair)`: Get count of open buy orders
- `get_sell_order_count(pair)`: Get count of open sell orders

### Price Data
- `get_bid_price(step)`: Get current bid price
- `get_ask_price(step)`: Get current ask price
- `get_last_trade_buy_price()`: Get the price of the last executed buy
- `get_last_trade_sell_price()`: Get the price of the last executed sell

### Time Management
- `get_current_step()`: Get the current timestep in the simulation

## Example Order Dictionary:

```python
order_dict = {
"pair": "BTC", # Trading pair
"side": "buy", # 'buy' or 'sell'
"price": 50000, # Order price
"amount": 1.0, # Order quantity
"step": step # Current timestep
}
```


## Example Strategy Implementation
In order to check an example for how to implement a strategy, you can check provided dca_v0.py file

    
## Best Practices
1. Always inherit from `BaseStrategy`
2. Implement the required `make_step_controls` method
3. Use the provided methods instead of accessing `self.backtest` directly
4. Handle both entry and exit conditions in your strategy
5. Properly manage open orders and positions
6. Use strategy parameters through `self.params` for easy configuration

## Note
The framework automatically handles:
- Backtesting environment setup
- Order execution
- Position tracking
- Performance metrics
- Data management
- Close position at the end of the backtest

You only need to focus on implementing your trading logic in the `make_step_controls` method.
