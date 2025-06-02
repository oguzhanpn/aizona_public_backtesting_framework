#%%
from datetime import datetime
import time
is_live = True
if is_live == False:
    from aizona_public_backtesting_framework.base_strategy import BaseStrategy
else:
    from aizona_quant_engine_v2.live_base_strategy import BaseStrategy


class Strategy(BaseStrategy):
    """Take Profit Strategy Implementation"""
    
    strategy_name = "TakeProfitStrategy"
    configs = {
        # Strategy parameters
        "params": {
            "pos_size": 1000,
            "take_profit_threshold": 0.001,
            "stop_loss_threshold": 0.01
        },
        
        # Backtest configuration
        "backtest_config": {
            "pair_list": ["BTC"],
            "start_date": datetime(2024, 1, 1),
            "end_date": datetime(2024, 12, 31),
            'resample_freq_in_ms': 1000,
            'use_ob_levels': 1
        }
    }

    def __init__(self, pair_list, params, live_configs=None):
        super().__init__(pair_list, params, strategy_id=self.strategy_name, live_configs=live_configs)

    def make_step_controls(self, pair):
                
        new_order_dict = self.get_new_order_dict(pair)

        if self.control_cancels(pair, new_order_dict):
            return
        
        if new_order_dict:
            self.create_limit_order(new_order_dict)

    def get_new_order_dict(self, pair):
        current_pos = self.get_position(pair)
        step = self.get_current_step()        
        profit = self.get_pos_profit_pct(pair)
        order_dict = None
        if abs(current_pos * self.get_mid_price(step, pair)) < 50:
            self.logger.info(f"Opening position {current_pos} - {self.params['pos_size']}")
            price = self.get_bid_price(step, pair)
            if price == 0:
                self.logger.info(f"Price is 0 for {pair} at step {step}")
                time.sleep(1)
                return None
            amount = self.params['pos_size'] / price
            order_dict = {
                "pair": pair, 
                "side": "sell", 
                "price": price, 
                "amount": amount,
                "step": step,
                "maker": True,
                "comment": "first_order"
            }

        elif (current_pos > 0 and 
                (profit > self.params['take_profit_threshold'] or profit < -self.params['stop_loss_threshold'])
                ):
            profit_or_loss = 'take profit' if profit > 0 else 'stop loss'
            self.logger.info(f"step: {step} {profit_or_loss} from long side {profit} - current pos {current_pos}")
            price = self.get_ask_price(step, pair)  
            if price == 0:
                self.logger.info(f"Price is 0 for {pair} at step {step}")
                return None
            amount = self.params['pos_size'] * 2 / price
            order_dict = {
                "pair": pair, 
                "side": "sell", 
                "price": price, 
                "amount": amount,
                "step": step,
                "maker": True,
                "comment": f"{profit_or_loss} from long side"
            }

        elif (current_pos < 0 and 
                (profit > self.params['take_profit_threshold'] or profit < -self.params['stop_loss_threshold'])
                ):
            profit_or_loss = 'take profit' if profit > 0 else 'stop loss'
            self.logger.info(f"step: {step} {profit_or_loss} from short side {profit} - current pos {current_pos}")
            price = self.get_bid_price(step, pair)
            if price == 0:
                self.logger.info(f"Price is 0 for {pair} at step {step}")
                return None
            amount = self.params['pos_size'] * 2 / price
            order_dict = {
                "pair": pair, 
                "side": "buy", 
                "price": price, 
                "amount": amount,
                "step": step,
                "maker": True,
                "comment": f"{profit_or_loss} from short side"
            }

        return order_dict

    def control_cancels(self, pair, new_order_dict):
        found_order = False
        orders = self.get_open_orders(pair)
        for order in orders:
            price = order.price
            amount = order.amount
            side = order.side
            self.logger.info(f"open order found: order_id: {order.order_id} price: {order.price} amount: {order.amount} side: {order.side}")
            if (
                new_order_dict 
                and not found_order 
                and price == new_order_dict['price'] 
                and amount == new_order_dict['amount'] 
                and side == new_order_dict['side']
                ):
                found_order = True
                self.logger.info(f"new order already exist in open orders: order_id: {order.order_id} price: {order.price} amount: {order.amount} side: {order.side}")
                continue
            
            self.cancel_open_order(order)

        return found_order

    def get_pos_profit_pct(self, pair):
        """Get the profit of the position"""
        current_position = self.get_position(pair)
        current_price = self.get_mid_price(self.get_current_step(), pair)
        pos_entry_price = self.get_open_position_price(pair)
        if current_price == 0:
            self.logger.info(f"Price is 0 for {pair} at step {self.get_current_step()}")
            return 0

        if current_position == 0:
            return 0

        pos_entry_val = current_position * pos_entry_price
        current_pos_val = current_position * current_price
        profit = current_pos_val - pos_entry_val
        return profit / abs(pos_entry_val)
    
  
if __name__ == "__main__":
    from runing_tools import run_strategy_on_server

    username = "" #Your username here
    password = "" #Your password here
    strategy_file_name = "take_profit_strategy.py"

    run_strategy_on_server(
        username=username,
        password=password,
        script_path=strategy_file_name,
        file_name=strategy_file_name
    )
