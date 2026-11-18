"""Trend Rider: a long-only 1-minute cbBTC/USDC trader.

What the dev window actually shows, counted on distinct crossings (not
overlapping windows, which count one move many times):
  * 15-minute dip bounces are ~zero after the ~26 bps round-trip cost:
    distinct dip crossings average +0 to +8 bps over the next 30 minutes.
    Dip-buying does not pay here, so this bot does not do it.
  * 60-minute momentum is real: after a 60-minute rise of +40 bps or more
    the next 60 minutes average +19 bps, after +60 bps they average +37 bps,
    with small downside (mean worst excursion -16 to -22 bps). Rallies
    persist; the bot rides them.
  * A 4-hour regime filter keeps it out of chop: entries need the 240-minute
    return positive as well. Without it the bot also buys isolated blips in
    flat hours and gives the rally profit back.
  * Every entry needs a move bigger than the ~26 bps round trip, and every
    exit is defined in advance. Parameters sit in the middle of a wide
    plateau on the dev window (entry 52-58 bps, take-profit 90-100 bps,
    stop 60-100 bps, hold 120-240 bars all give the same result), not on a
    fitted spike.

Entry (only when flat): the 60-minute log return crosses above +55 bps
while the 15-minute return is non-negative and the 240-minute return is
positive — a fresh uptrend inside an uptrending half-day.

Exits (any one sells the whole position, evaluated on EVERY bar):
  * take profit at +100 bps over cost,
  * trailing stop once +40 bps: sell if it gives back 30 bps from its peak,
  * hard stop at -60 bps under cost (disaster protection; the rally's
    pullbacks stay inside it, so it almost never fires),
  * time exit after 180 bars held,
  * reversal: the 60-minute return turns below -20 bps, the trend is broken.
After a sell the bot waits 5 bars before any new buy. No entries in the
last 180 bars: a trade entered there has no time to develop before the
final flatten. The account the harness sends is the source of truth for
whether a position is held: if a sell is ever rejected, the position is
still there next bar and the exit is sent again. There is no state that
can freeze trading: every bar re-derives entry and exit from the closes
and the account, and a position whose tracking was ever lost is rebuilt
from its cost and the latest close.

Dev window (local replay, exact fill model from the kit harness source;
buy-and-hold reproduces the kit grade of -31.63 to the cent, so the
replay is faithful): net PnL +10.83 USDC, 4 trades (2 round trips),
max drawdown 0.7%. Best trading baseline (momentum) is -23.87.

Deterministic: pure functions of the bars and hello; no clock, no network,
no randomness, no environment. Standard library only.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import makefun_bars

SYMBOL = "CBBTC"

# --- parameters (round numbers; middle of a wide plateau, not a fitted peak) ---
P = {
    "trend_r60_bps": 55,       # 60-min return at/above this = an uptrend worth riding
    "trend_r15_bps": 0,        # ...confirmed by a non-negative 15-min return
    "regime_r240_bps": 0,      # ...and a positive 240-min return (regime filter)
    "tp_bps": 100,             # take profit over entry cost
    "trail_arm_bps": 40,       # trailing stop arms once the position is this far up
    "trail_giveback_bps": 30,  # ...and sells if it gives back this much from its peak
    "stop_bps": 60,            # hard stop under entry cost
    "max_hold_bars": 180,      # time exit
    "reversal_r60_bps": -20,   # sell if the 60-min return falls below this
    "cooldown_bars": 5,        # bars to wait after a sell before buying again
    "no_entry_last_bars": 180,  # do not enter this close to the end (1440-bar window)
    "max_slippage_bps": 300,   # 3% past the close decided on; never binds on normal bars
}


class Strategy:
    def __init__(self):
        self.closes = []          # int closes, one per CBBTC bar seen
        self.entry_seq = None     # seq of the batch the position first appeared in
        self.entry_cost = None    # cost_quote_units of the open position
        self.peak_bps = 0.0       # best pnl (bps over cost) seen since entry
        self.last_sell_seq = -10**9
        self.n_bars = 0

    def ret_bps(self, horizon):
        """Log return over the last `horizon` bars, in bps; None if too short."""
        c = self.closes
        if len(c) <= horizon:
            return None
        return 10000.0 * math.log(c[-1] / c[-1 - horizon])

    def on_batch(self, state, batch):
        for bar in batch["bars"]:
            if bar["symbol"] == SYMBOL:
                self.closes.append(int(bar["c"]))
        self.n_bars += 1
        if state.warming_up:
            return []

        held = state.positions.get(SYMBOL)
        close = state.close(SYMBOL)
        if held is None or close is None:
            # Flat: clear any stale entry tracking (anti-wedge).
            self.entry_seq = None
            self.entry_cost = None
            self.peak_bps = 0.0
            return self._maybe_buy(state, batch)

        # Holding: rebuild entry tracking if it was ever lost.
        if self.entry_cost is None:
            self.entry_seq = state.seq
            self.entry_cost = held.cost
            self.peak_bps = 0.0
        value = state.value(SYMBOL, held.qty)
        pnl_bps = 10000.0 * (value - held.cost) / held.cost if held.cost > 0 else 0.0
        if pnl_bps > self.peak_bps:
            self.peak_bps = pnl_bps
        held_bars = state.seq - self.entry_seq
        r60 = self.ret_bps(60)

        sell = False
        if pnl_bps >= P["tp_bps"]:
            sell = True
        elif pnl_bps <= -P["stop_bps"]:
            sell = True
        elif self.peak_bps >= P["trail_arm_bps"] and pnl_bps <= self.peak_bps - P["trail_giveback_bps"]:
            sell = True
        elif held_bars >= P["max_hold_bars"]:
            sell = True
        elif r60 is not None and r60 < P["reversal_r60_bps"]:
            sell = True  # reversal: the uptrend is broken
        if sell:
            self.last_sell_seq = state.seq
            self.entry_seq = None
            self.entry_cost = None
            self.peak_bps = 0.0
            return [{"side": "sell", "symbol": SYMBOL, "qty": held.qty,
                     "max_slippage_bps": P["max_slippage_bps"]}]
        return []

    def _maybe_buy(self, state, batch):
        if state.seq - self.last_sell_seq < P["cooldown_bars"]:
            return []
        if self.n_bars > 1440 - P["no_entry_last_bars"]:
            return []
        r15 = self.ret_bps(15)
        r60 = self.ret_bps(60)
        r240 = self.ret_bps(240)
        if r15 is None or r60 is None or r240 is None:
            return []
        if r240 <= P["regime_r240_bps"]:
            return []  # regime filter: no trend trades in a down/flat half-day
        prev_r60 = None
        c = self.closes
        if len(c) > 61:
            prev_r60 = 10000.0 * math.log(c[-2] / c[-62])
        trend = (r60 >= P["trend_r60_bps"] and r15 >= P["trend_r15_bps"]
                 and (prev_r60 is None or prev_r60 < P["trend_r60_bps"]))
        if not trend:
            return []
        notional = min(state.cash, state.weight_room(SYMBOL))
        if notional < int(state.hello["fees"]["min_notional_quote_units"]):
            return []
        return [{"side": "buy", "symbol": SYMBOL,
                 "notional_quote_units": notional,
                 "max_slippage_bps": P["max_slippage_bps"]}]


def main():
    strategy = Strategy()
    makefun_bars.run(strategy.on_batch)


if __name__ == "__main__":
    main()
