Agents desc:

Random agent (random_agent.py)
-fully random
-used to compare other agent strategies


Momentum trading strat (momentum_agent.py)
-uses a momentum strategy
-buying high and selling higher
-does not work well now because no event shocks (will add later)

Mean reversion strat (mean_reversion_agents.py)
If price is below recent average by enough -> buy
If price is above recent average by enough -> sell
Otherwise -> hold

Buy and hold strat (buy_and_hold_agent.py)
buy once at beggingn and then hold

Contrarian agent (contrarian_agent.py)
opposite of momentum
if price dropped sharply last step -> buy (expects bounce)
if price rose sharply last step -> sell (expects pullback)
does nothing if the move was small

Volume pressure agent (volume_pressure_agent.py)
watches order flow, not price
if recent buying pressure was strong relative to volume -> buy if recent selling pressure was strong relative to volume -> sell
ignores price direction entirely

Value estimate agent (value_estimate_agent.py)
estimates fair value from a long moving average of price history
if current price is well below that estimate -> buy
if current price is well above that estimate -> sell
slower and more patient than mean reversion, reacts to bigger dislocations

Risk managed agent (risk_managed_agent.py)
mean reversion strategy underneath
adds hard limits on top: max position size, reduces trade size when volatility is high, stops buying if portfolio is down too much


RL agent (rl_agent.py)
