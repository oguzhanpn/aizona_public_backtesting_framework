from abc import ABC, abstractmethod
import requests
import logging
from datetime import datetime
from pathlib import Path
import sys
class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    All strategy implementations must inherit from this class and implement its abstract methods.
    """
    def __init__(self, pair_list, params):
        self.log_name = 'strategy_logger'
        Path("logs").mkdir(parents=True, exist_ok=True)
        self.log_file = 'logs/' + datetime.now().strftime('%Y-%m-%d %H-%M-%S') + '_' + self.log_name + '.log'
        self._setup_logger(self.log_name)
        self.logger = logging.getLogger(self.log_name)
        self.pair_list = pair_list
        self.params = params
        self.backtest = None
        self.sim_size = None
        self.results_channel = None

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

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def _close_pair_positions(self, pairs, comment='close_pos'):
        for pair in pairs:
            # First cancel open orders:
            open_orders = self.backtest.order_manager.get_orders(pair)
            for _order in open_orders:
                self.backtest.order_manager.cancel_open_order(_order)

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
        # self.backtest.metric_object.get_cumulative_pnl_plot()
        self.backtest.metric_object.send_cumulative_pnl_plot(dc_url=self.results_channel)

    def run_strategyy(self):
        while self.backtest.run:
            pair = self.backtest.get_data("pair", self.backtest.current_step)
            self.strategy_manager_pair(pair)
            self.backtest.pass_steps()

        if self.backtest.reset_at_end:
            for pair in self.pair_list:
                self.backtest.order_manager.cancel_open_orders(pair)
                self.backtest.control_buy_orders(pair)
                self.backtest.control_sell_orders(pair)

            self._close_pair_positions(self.pair_list)
            self.backtest.trade_history.expectancy()

        self._send_backtest_report()

    def get_current_step(self):
        return self.backtest.current_step
    
    def get_buy_order_count(self, pair):
        return self.backtest.order_manager.get_buy_order_count(pair)
    
    def get_sell_order_count(self, pair):
        return self.backtest.order_manager.get_sell_order_count(pair)
    
    def get_position(self, pair):
        return self.backtest.trade_history.positions[pair]
    
    def get_bid_price(self, step):
        return self.backtest.price_data.get_data("bid_0_price", step)
    
    def get_ask_price(self, step):
        return self.backtest.price_data.get_data("ask_0_price", step)
    
    def get_mid_price(self, step):
        return (self.get_ask_price(step) + self.get_bid_price(step)) / 2

    def get_data(self, data_type, step):
        return self.backtest.price_data.get_data(data_type, step)
    
    def get_last_trade_buy_price(self):
        return self.backtest.trade_history.last_trade_buy.price
    
    def get_last_trade_sell_price(self):
        return self.backtest.trade_history.last_trade_sell.price
    
    def get_open_position_price(self, pair):
        return self.backtest.trade_history.open_positions_price[pair]
    
    def create_limit_order(self, order_dict):
        self.backtest.create_limit_order(order_dict)
    
    def get_buy_orders(self, pair):
        return self.backtest.order_manager.get_buy_orders(pair)
    
    def get_sell_orders(self, pair):
        return self.backtest.order_manager.get_sell_orders(pair)
    
    def cancel_open_order(self, order):
        self.backtest.order_manager.cancel_open_order(order)

    def cancel_open_orders(self, pair):
        self.backtest.order_manager.cancel_open_orders(pair)

    def printer_method(self, message: str):
        requests.post(self.results_channel, {"content": message})


