from aizona_strategies.base_strategy import BaseStrategy


class strategy(BaseStrategy):
    strategy_name="dca_v0"

    def __init__(self, pair_list, params, data_dir_name=None):
        super().__init__(pair_list, params, self.strategy_name, data_dir_name)

    def control_buy_limit_order(self, pair):
        step = self.get_current_step()
        if self.get_buy_order_count(pair) == 0:
            if self.get_position(pair) == 0:
                if self.params['side'] == "LONG":
                    # put start order
                    price = self.get_bid_price(step)
                    _dict = {"pair": pair, "side": "buy", "price": price, "amount": self.params['order_amount']/price, "step": step}
                    self.create_limit_order(_dict)
                else:
                    # wait do nothing - to be implemented..
                    pass
            else:
                if self.params['side'] == "LONG":
                    # add dca order
                    price = self.get_last_trade_buy_price() * (1-self.params['price_deviation_to_safety_order']/100)
                    _dict = {"pair": pair, "side": "buy", "price": price, "amount": self.params['order_amount']/price, "step": step}
                    self.create_limit_order(_dict)
                else:
                    #close pos order
                    price = self.get_open_position_price(pair) * (1-self.params['take_profit']/100)
                    _dict = {"pair": pair, "side": "buy", "price": price, "amount": self.get_position(pair), "step": step}
                    self.create_limit_order(_dict)

    def control_sell_limit_order(self, pair):
        step = self.get_current_step()
        if self.get_sell_order_count(pair) == 0:
            if self.get_position(pair) == 0:
                if self.params['side'] == "SHORT":
                    # put start order
                    price = self.get_ask_price(step)
                    _dict = {"pair": pair, "side": "sell", "price": price, "amount": self.params['order_amount']/price, "step": step}
                    self.create_limit_order(_dict)
                else:
                    # wait do nothing
                    pass
            else:
                if self.params['side'] == "SHORT":
                    # add dca order
                    price = self.get_last_trade_sell_price() * (1+self.params['price_deviation_to_safety_order']/100)
                    _dict = {"pair": pair, "side": "sell", "price": price, "amount": self.params['order_amount']/price, "step": step}
                    self.create_limit_order(_dict)
                else:
                    #close pos order
                    price = self.get_open_position_price(pair) * (1+self.params['take_profit']/100)
                    _dict = {"pair": pair, "side": "sell", "price": price, "amount": self.get_position(pair), "step": step}
                    self.create_limit_order(_dict)

    def cancel_control_buy_limit_order(self, pair):
        step = self.get_current_step()
        buy_orders = self.get_buy_orders(pair)
        
        if len(buy_orders) != 0:
            if self.get_position(pair) == 0:
                if step - buy_orders[0].step > 60:
                    self.cancel_open_orders(pair)

    def cancel_control_sell_limit_order(self, pair):
        step = self.get_current_step()
        sell_orders = self.get_sell_orders(pair)
        
        if len(sell_orders) != 0:
            if self.get_position(pair) == 0:
                if step - sell_orders[0].step > 60:
                    self.cancel_open_orders(pair)

    def make_step_controls(self, pair):
        self.control_buy_limit_order(pair)
        self.control_sell_limit_order(pair)
        self.cancel_control_buy_limit_order(pair)
        self.cancel_control_sell_limit_order(pair)


if __name__=="__main__":

    params={
        "side": "LONG",
        "order_amount": 1000,
        "take_profit": 0.2,
        "price_deviation_to_safety_order": 0.1}

    self=strategy(
        pair_list = ["BTC"],
        params=params,
        data_dir_name='tardis_data_test/'
    )
    self.run_strategyy()




