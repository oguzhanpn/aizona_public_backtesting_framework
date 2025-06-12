from abc import ABC, abstractmethod
import requests
import logging
from datetime import datetime
from pathlib import Path
import sys
import ccxt
import traceback
class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    All strategy implementations must inherit from this class and implement its abstract methods.
    """
    def __init__(self, pair_list, params, strategy_id=None):
        self.log_name = 'strategy_logger'
        Path("logs").mkdir(parents=True, exist_ok=True)
        self.log_file = 'logs/strategy_logs/' + datetime.now().strftime('%Y-%m-%d %H-%M-%S') + '_' + self.log_name + '.log'
        self._setup_logger(self.log_name)
        self.logger = logging.getLogger(self.log_name)
        self.pair_list = pair_list
        self.params = params
        self.backtest = None
        self.sim_size = None
        self.results_channel = None
        self.next_progress_log_percentage = 0.1
        self.strategy_id = strategy_id
        self.last_issued_trade = None
    
    @abstractmethod
    def make_step_controls(self, pair):
        """
        Perform step-by-step control checks for the strategy.
        Must be implemented by concrete strategy classes.
        """
        raise NotImplementedError("make_step_controls method must be implemented") 
    
    def strategy_manager_pair(self, pair):
        self.make_step_controls(pair)

    def _setup_logger(self, name):
        """Set up a logger with the given name, logging to a file and console."""
        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.log_file)
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def _close_pair_positions(self, pairs, comment='close_pos'):
        for pair in pairs:
            # First cancel open orders:
            open_orders = self.get_open_orders(pair)
            for _order in open_orders:
                self.cancel_open_order(_order)

            # Now close all positions:
            pos_amount = self.backtest.get_position(pair)
            if pos_amount == 0:
                continue

            elif pos_amount > 0:
                best_bid = self.backtest.get_pair_best_bid(pair)
                new_order_dict = {"pair": pair,
                                  "side": 'sell',
                                  "price": best_bid,
                                  "amount": abs(pos_amount),
                                  "step": self.backtest.current_step,
                                  "maker": False,
                                  "comment": comment}

            else:
                best_ask = self.backtest.get_pair_best_ask(pair)
                new_order_dict = {"pair": pair,
                                  "side": 'buy',
                                  "price": best_ask,
                                  "amount": abs(pos_amount),
                                  "step": self.backtest.current_step,
                                  "maker": False,
                                  "comment": comment}

            self.backtest.create_limit_order(new_order_dict)

    def _send_backtest_report(self):
        self.backtest.create_backtest_report(self.results_channel)
        self.backtest.metric_object.send_cumulative_pnl_plot(dc_url=self.results_channel)

    def _send_progress_log_if_necessary(self):
        completion_ratio = self.backtest.current_step / self.backtest.max_steps
        if completion_ratio > self.next_progress_log_percentage:
            print('completion_ratio:', completion_ratio)
            self.printer_method(f"Backtest is {int(completion_ratio * 100)}% complete")
            self.next_progress_log_percentage += 0.1

    def run_strategyy(self):
        while self.backtest.run:
            pair = self.backtest.get_data("pair", self.backtest.current_step)
            self.strategy_manager_pair(pair)
            self.backtest.pass_steps()
            self._send_progress_log_if_necessary()

        if self.backtest.reset_at_end:
            for pair in self.pair_list:
                self.backtest.order_manager.cancel_open_orders(pair)
                self.backtest.control_buy_orders(pair)
                self.backtest.control_sell_orders(pair)

            self._close_pair_positions(self.pair_list)
            self.backtest.trade_history.expectancy()

            self._send_backtest_report()

    def pass_step_from_strategy(self):
        if self.get_current_step() < self.backtest.max_steps - 1:
            self.backtest.pass_steps()
            return True
        else:
            return False

    def get_sleep_multiplier(self):
        return 0
    
    def get_current_step(self):
        return self.backtest.current_step
    
    def get_buy_order_count(self, pair):
        return self.backtest.order_manager.get_buy_order_count(pair)
    
    def get_sell_order_count(self, pair):
        return self.backtest.order_manager.get_sell_order_count(pair)
    
    def get_position(self, pair):
        return self.backtest.trade_history.positions[pair]
    
    def get_old_position(self):
        return self.backtest.trade_history.old_pos_for_peak_control

    def get_bid_price(self, step, pair=None):
        return self.backtest.price_data.get_data("bid_0_price", step)
    
    def get_ask_price(self, step, pair=None):
        return self.backtest.price_data.get_data("ask_0_price", step)
    
    def get_mid_price(self, step, pair=None):
        return (self.get_ask_price(step) + self.get_bid_price(step)) / 2

    def get_data(self, data_type, step):
        return self.backtest.price_data.get_data(data_type, step)

    def get_historical_data(self, step, lookback_in_seconds, pair):
        return self.backtest.price_data.get_historical_data(step, lookback_in_seconds, pair)
    
    def get_prediction(self, step, model_name):
        return self.backtest.get_prediction(step, model_name)
    
    def make_future_predictions(self, model_name, model_object, pair=None):
        self.backtest.make_future_predictions(model_name, model_object, pair)

    def get_last_trade_buy_price(self, pair):
        return self.backtest.trade_history.last_trade_buy[pair].price
    
    def get_last_trade_sell_price(self, pair):
        return self.backtest.trade_history.last_trade_sell[pair].price
    
    def get_last_trade(self, pair):
        if self.backtest.trade_history.trade_objects_list:
            return self.backtest.trade_history.trade_objects_list[-1]
        else:
            return None

    def get_my_trades(self, pair=None):
        if pair is None:
            return self.backtest.trade_history.trade_objects_list
        else:
            return [trade for trade in self.backtest.trade_history.trade_objects_list if trade.pair == pair]
        
    def get_new_trades(self):
        last_trade = self.get_last_trade(self.pair_list[0])
        if last_trade == self.last_issued_trade:
            return []

        if self.last_issued_trade is None:
            new_trades = self.get_my_trades()
        else:
            new_trades = [trade for trade in self.get_my_trades() if trade.timestamp > self.last_issued_trade.timestamp]
        if new_trades:
            self.update_last_issued_trade(new_trades)
        return new_trades
    
    def update_last_issued_trade(self, new_trades):
        self.last_issued_trade = max(new_trades, key=lambda trade: trade.timestamp)

    def get_min_order_amounts(self):
        markets = ccxt.binance({
            'options': {
                'defaultType': 'future'
            }
        }).load_markets()
        min_order_amounts ={}
        for pair in self.pair_list:
            symbol = pair + "/USDT:USDT"
            # Get market info
            market = markets[symbol]
            # Extract minimum order amount
            min_order_amounts[pair] = market['limits']['amount']['min']
        return min_order_amounts

    def get_open_position_trades(self, pair):
        return self.backtest.trade_history.temp_trades_list[pair]

    def get_open_position_price(self, pair):
        return self.backtest.trade_history.open_positions_price[pair]
    
    def create_limit_order(self, order_dict):
        return self.backtest.create_limit_order(order_dict)

    def get_open_orders(self, pair):
        return self.backtest.order_manager.get_orders(pair)
    
    def get_buy_orders(self, pair):
        return self.backtest.order_manager.get_buy_orders(pair)
    
    def get_sell_orders(self, pair):
        return self.backtest.order_manager.get_sell_orders(pair)
    
    def cancel_open_order(self, order):
        self.backtest.order_manager.cancel_open_order(order)

    def cancel_open_orders(self, pair):
        self.backtest.order_manager.cancel_open_orders(pair)

    def control_trades(self):
        return self.backtest.control_trades()

    def get_instant_pnl(self, pair):
        return self.backtest.trade_history.instant_pnl[pair]

    def get_trade_executed(self, pair):
        return self.backtest.trade_executed[pair]

    def set_trade_executed(self, pair, value):
        self.backtest.trade_executed[pair] = value

    def reset_trade_history(self):
        self.backtest.reset_trade_history()

    def update_trade_history(self, trade_infos):
        self.backtest.update_trade_history(trade_infos)

    def get_max_steps(self):
        return self.backtest.max_steps

    def printer_method(self, message: str):
        requests.post(
            url=self.results_channel,
            json={"content": message}
        )

