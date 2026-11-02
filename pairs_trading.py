# pairs trading project - ben
# finds cointegrated pairs on 2016-2022 data then backtests on 2023-2025

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

pd.set_option("display.width", 120)
os.chdir(os.path.dirname(os.path.abspath("pairs-trading")))

if not os.path.exists("data"):
    os.mkdir("data")
if not os.path.exists("results"):
    os.mkdir("results")

pairs = {
    "UK banks": ("LLOY.L", "NWG.L"),
    "Oil majors": ("BP.L", "SHEL.L"),
    "Supermarkets": ("TSCO.L", "SBRY.L"),
    "Insurers": ("AV.L", "LGEN.L"),
    "Miners": ("RIO.L", "AAL.L"),
    "Country ETFs": ("EWA", "EWC"),
}

entry = 2
exit_level = 0.5
cost = 0.001

tickers = []
for p in pairs.values():
    tickers.append(p[0])
    tickers.append(p[1])

# only download if i havent already
if os.path.exists("data/prices.csv"):
    prices = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
else:
    data = yf.download(tickers, start="2016-01-01", end="2026-01-01", auto_adjust=True, progress=False)
    prices = data["Close"]
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.to_csv("data/prices.csv")

print(prices.shape)

# check data
for t in prices.columns:
    first = prices[t].first_valid_index()
    biggest = prices[t].pct_change(fill_method=None).abs().max() * 100
    print(t, first.date(), "missing:", prices[t].isna().sum(), "biggest move:", round(biggest, 1), "%")

# split into train and test
train = {}
test = {}
for name in pairs:
    a = pairs[name][0]
    b = pairs[name][1]
    df = prices[[a, b]].dropna()
    if len(df) == 0 or df.index[0] > pd.Timestamp("2016-01-15"):
        print("dropping", name, "not enough history")
        continue
    train[name] = df.loc[:"2022-12-31"]
    test[name] = df.loc["2023-01-01":]
    print(name, len(train[name]), len(test[name]))

# plot prices
for name in train:
    df = pd.concat([train[name], test[name]])
    df = df / df.iloc[0] * 100
    df.plot(figsize=(10, 4), title=name)
    plt.axvline(pd.Timestamp("2023-01-01"), color="k", linestyle="--")
    plt.savefig("results/prices_" + name.replace(" ", "_") + ".png")
    plt.close()

# cointegration test
print("\ncointegration (train data)")
keep = []
table = []
for name in train:
    df = train[name]
    a = df.columns[0]
    b = df.columns[1]
    r = df.pct_change().dropna()
    corr = r[a].corr(r[b])
    score, pval, _ = coint(np.log(df[a]), np.log(df[b]))
    print(name, "corr =", round(corr, 2), "p =", round(pval, 3))
    table.append([name, round(corr, 2), round(pval, 3), pval < 0.05])
    if pval < 0.05:
        keep.append(name)

pd.DataFrame(table, columns=["pair", "corr", "p value", "passes"]).to_csv("results/pair_selection.csv", index=False)
print("keeping:", keep)

if len(keep) == 0:
    print("no pairs passed")
    raise SystemExit

# hedge ratio, mean, std from train
hedge = {}
means = {}
stds = {}
for name in keep:
    df = train[name]
    a = df.columns[0]
    b = df.columns[1]
    X = sm.add_constant(np.log(df[b]))
    model = sm.OLS(np.log(df[a]), X).fit()
    hedge[name] = model.params[b]
    spread = np.log(df[a]) - hedge[name] * np.log(df[b])
    means[name] = spread.mean()
    stds[name] = spread.std()
    print(name, "beta", round(hedge[name], 4), "mean", round(means[name], 4), "std", round(stds[name], 4))


def get_positions(z):
    pos = []
    current = 0
    for val in z:
        if current == 0:
            if val > entry:
                current = -1
            elif val < -entry:
                current = 1
        else:
            if abs(val) < exit_level:
                current = 0
        pos.append(current)
    return pd.Series(pos, index=z.index)


def run_backtest(name, c):
    df = test[name]
    a = df.columns[0]
    b = df.columns[1]
    spread = np.log(df[a]) - hedge[name] * np.log(df[b])
    z = (spread - means[name]) / stds[name]
    pos = get_positions(z)
    held = pos.shift(1).fillna(0)  # trade the day after the signal
    pnl = (held * spread.diff()).fillna(0)
    pnl = pnl - held.diff().abs().fillna(0) * c * 2
    return z, pos, pnl


def count_trades(pos):
    return int(((pos != 0) & (pos.shift(1).fillna(0) == 0)).sum())


def stats(returns, n_trades):
    eq = (1 + returns).cumprod()
    total = (eq.iloc[-1] - 1) * 100
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    dd = (eq / eq.cummax() - 1).min() * 100
    return [round(total, 2), round(sharpe, 2), round(dd, 2), n_trades]


print("\nbacktest 2023-2025")
all_pnl = {}
all_pnl_nocost = {}
all_z = {}
all_pos = {}
rows = []
total_trades = 0
for name in keep:
    z, pos, pnl = run_backtest(name, cost)
    z2, pos2, pnl2 = run_backtest(name, 0)
    all_pnl[name] = pnl
    all_pnl_nocost[name] = pnl2
    all_z[name] = z
    all_pos[name] = pos
    n = count_trades(pos)
    total_trades += n
    rows.append([name] + stats(pnl, n))

per_pair = pd.DataFrame(rows, columns=["pair", "total return %", "sharpe", "max drawdown %", "trades"])
print(per_pair)
per_pair.to_csv("results/per_pair_results.csv", index=False)

port = pd.DataFrame(all_pnl).fillna(0).mean(axis=1)
port_nocost = pd.DataFrame(all_pnl_nocost).fillna(0).mean(axis=1)

summary = pd.DataFrame([["with costs"] + stats(port, total_trades),
                        ["no costs"] + stats(port_nocost, total_trades)],
                       columns=["portfolio", "total return %", "sharpe", "max drawdown %", "trades"])
print(summary)
summary.to_csv("results/portfolio_results.csv", index=False)

# z score chart for first pair
name = keep[0]
z = all_z[name]
pos = all_pos[name]
opened = pos[(pos != 0) & (pos.shift(1).fillna(0) == 0)].index
closed = pos[(pos == 0) & (pos.shift(1).fillna(0) != 0)].index

plt.figure(figsize=(10, 4))
plt.plot(z, label="z score")
plt.axhline(2, color="grey", linestyle="--")
plt.axhline(-2, color="grey", linestyle="--")
plt.axhline(0, color="grey", linestyle="--")
plt.scatter(opened, z[opened], color="red", label="open")
plt.scatter(closed, z[closed], color="black", marker="x", label="close")
plt.title(name + " z score (test period)")
plt.legend()
plt.savefig("results/zscore_trades.png", dpi=150)
plt.close()

# equity curve
plt.figure(figsize=(10, 4))
plt.plot((1 + port).cumprod(), label="with costs")
plt.plot((1 + port_nocost).cumprod(), label="no costs", linestyle="--")
plt.axhline(1, color="black")
plt.title("growth of 1 pound, test period")
plt.legend()
plt.savefig("results/equity_curve.png", dpi=150)
plt.close()

print("done")
