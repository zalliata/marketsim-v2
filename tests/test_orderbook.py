from marketsim.engine.orderbook import OrderBook
from marketsim.types import Order, OrderType, Side


def test_price_time_priority_and_fill():
    ob = OrderBook("SIVB")
    ob.submit(Order("mm", "SIVB", Side.SELL, 100, OrderType.LIMIT, 100.0), tick=0)
    ob.submit(Order("mm", "SIVB", Side.SELL, 100, OrderType.LIMIT, 101.0), tick=0)
    trades = ob.submit(Order("buyer", "SIVB", Side.BUY, 150, OrderType.MARKET), tick=1)
    # fills cheapest first: 100 @100 then 50 @101
    assert [(t.price, t.quantity) for t in trades] == [(100.0, 100), (101.0, 50)]
    assert ob.best_ask() == 101.0
    assert ob.depth(Side.SELL) == 50


def test_no_self_trade():
    ob = OrderBook("SIVB")
    ob.submit(Order("a", "SIVB", Side.SELL, 100, OrderType.LIMIT, 100.0), tick=0)
    trades = ob.submit(Order("a", "SIVB", Side.BUY, 100, OrderType.MARKET), tick=1)
    assert trades == []            # cannot trade against own resting order


def test_limit_rests_when_no_cross():
    ob = OrderBook("SIVB")
    ob.submit(Order("b", "SIVB", Side.BUY, 50, OrderType.LIMIT, 99.0), tick=0)
    assert ob.best_bid() == 99.0
    assert ob.mid() == 99.0        # only one side


def test_cancel():
    ob = OrderBook("SIVB")
    o = Order("b", "SIVB", Side.BUY, 50, OrderType.LIMIT, 99.0)
    ob.submit(o, tick=0)
    assert ob.cancel(o.order_id) is True
    assert ob.best_bid() is None
    assert ob.cancellations_this_tick == 1
