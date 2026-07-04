from marketsim.agents.rl_market_maker import discretize, QLearningMM


def test_discretize_buckets():
    s = discretize(-6, -0.7, -0.5, 0.05)
    assert s == "crash|very_bearish|short|high"
    assert discretize(0.0, 0.0, 0.0, 0.005) == "flat|neutral|neutral|low"


def test_q_update_moves_toward_reward():
    mm = QLearningMM(agent_id="P3", name="RL", agent_type="pricing_rl",
                     initial_cash=1e6, seed=1,
                     config={"training": True, "symbol": "SIVB"})
    mm._last = ("flat|neutral|neutral|low", "buy")
    before = mm._qrow("flat|neutral|neutral|low")["buy"]
    mm._update(reward=10.0, new_state="up|bullish|long|low")
    after = mm._qrow("flat|neutral|neutral|low")["buy"]
    assert after > before          # positive reward raises Q(s,buy)
