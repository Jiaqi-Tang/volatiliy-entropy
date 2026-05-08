# Data Source

Raw data source: [HistData's MetaTrader EUR/USD 1-minute bar data](https://www.histdata.com/download-free-forex-historical-data/?/metatrader/1-minute-bar-quotes/EURUSD).

Asset: EUR/USD.

Raw frequency: 1-minute OHLC bars.

Raw files used: calendar years 2016 through 2025.

Observed cleaned timestamp range: 2016-01-03 22:00 UTC to 2025-12-31 21:57 UTC.

Raw fields: timestamp, open bid, high bid, low bid, close bid, volume.

## Raw Observations

Let the raw 1-minute price bars be indexed by observed timestamps:

$$
t \in \mathcal{T}_{1m}
$$

For each timestamp $t$, the raw observation contains:

$$
(O_t, H_t, L_t, C_t, V_t)
$$

The data source states that all timestamps use Eastern Standard Time without DST. This project
interprets raw timestamps as fixed UTC-05:00, with no daylight-saving adjustment.
All analysis timestamps are converted to UTC.

The raw files contain missing observations. Missingness is not repaired by
forward-filling or interpolation.

## Caveats

These are treated as vendor data artifacts.

**Zeros in Volume field**

The volume field $V_t$ is not used, as $V_t=0$ for all raw observations.

**Daylight saving time**

HistData's files are intended to be fixed EST, but from 2019 onward some files show
an EU daylight-saving-time transition artifact around 19:00 file time.

Observed issues:

- EU DST-end duplicate rows appear from 19:00 through 19:59 in 2019, 2020, 2021,
  2022, 2023, and 2025.
- These duplicate rows are exact duplicates, so they are removed by exact-row
  deduplication.
- EU DST-start has a missing 19:00 through 19:59 hour from 2019 onward.
- 2024 has the EU DST-start missing-hour pattern but does not have the corresponding
  EU DST-end duplicate-hour pattern.

---

# Preprocessing

Objective: Transform raw 1-minute EUR/USD OHLC data into a clean 5-minute log-return series suitable for volatility and entropy analysis.

## Data cleaning

**Load Raw Data**

Load all raw EUR/USD MetaTrader CSV files from 2016 through 2025.

The initial raw dataset is:

$$
\mathcal{X}_{1m}^{raw}
= \{(t_i, O_i, H_i, L_i, C_i, V_i)\}_{i=1}^{N_{raw}}
$$

**Timestamp Interpretation**

Each raw timestamp is converted to UTC:

$$
t_i^{UTC} = t_i^{raw} + 5\text{ hours}
$$

The preprocessing does not apply daylight-saving-time shifts.

**Exact Deduplication**

An exact duplicate means the full raw observation is repeated:

$$
(t_i, O_i, H_i, L_i, C_i, V_i)
=
(t_j, O_j, H_j, L_j, C_j, V_j)
$$

for $i \neq j$.

If duplicate timestamps with different OHLC or volume values are found, this is treated as a data-quality error (rather than choosing one observation arbitrarily). No such duplications were found.

The cleaned 1-minute dataset is:

$$
\mathcal{X}_{1m}^{clean}
=
\{(t_i, O_i, H_i, L_i, C_i, V_i)\}_{i=1}^{N_{1m}}
$$

where timestamps are in UTC.

## Data aggregations

**Aggregate to 5-Minute OHLC Bars**

Let $B_j$ be the set of observed 1-minute bars whose timestamps fall inside the
5-minute interval indexed by $j$.

For every nonempty 5-minute interval:

$$
|B_j| \in \{1,2,3,4,5\}
$$

the 5-minute OHLC bar is defined as:

$$
O_j^{5m} = \text{first observed } O_i \text{ in } B_j
$$

$$
H_j^{5m} = \max_{i \in B_j} H_i
$$

$$
L_j^{5m} = \min_{i \in B_j} L_i
$$

$$
C_j^{5m} = \text{last observed } C_i \text{ in } B_j
$$

The number of observed 1-minute bars used in each 5-minute bar is:

$$
n_j^{1m} = |B_j|
$$

Every nonempty 5-minute bar is kept, including bars constructed from only 1, 2, 3,
or 4 observed 1-minute bars. Empty 5-minute bars are dropped.

The resulting 5-minute OHLC dataset is:

$$
\mathcal{X}_{5m}
=
\{(t_j, O_j^{5m}, H_j^{5m}, L_j^{5m}, C_j^{5m}, n_j^{1m})\}_{j=1}^{N_{5m}}
$$

**Compute Log Returns**

Let the observed 5-minute close price be:

$$
S_j = C_j^{5m},\quad \text{ where } S_j > 0
$$

For consecutive observed 5-minute timestamps, compute:

$$
r_j = \log(S_j) - \log(S_{j-1})
$$

The elapsed time for each candidate return is:

$$
\Delta t_j = t_j - t_{j-1}
$$

**Gap Filtering**

The expected gap is:

$$
\Delta t_{expected} = 5\text{ minutes}
$$

The final clean return series keeps only returns satisfying:

$$
\Delta t_j = \Delta t_{expected} = 5\text{ minutes}
$$

This strict rule avoids including weekend gaps, holiday gaps, outages, and
missing-candle jumps as ordinary 5-minute returns.

The final clean return series is:

$$
R = \{r_1, r_2, \ldots, r_N\}
$$

of consecutive log returns on 5m data.

---

## Output datasets

Preprocessing outputs are intermediate datasets and can be found in the
`data/intermediate` folder:

```text
data/intermediate/eurusd_1m_utc_clean.csv
data/intermediate/eurusd_5m_ohlc_utc_nonempty.csv
data/intermediate/eurusd_5m_log_returns_clean.csv
data/intermediate/preprocessing_report.json
```

The clean 5-minute return dataset contains `timestamp_utc, close, log_return, previous_timestamp_utc, previous_close, delta_minutes, n_m1`, where `n_m1` records the number of observed 1-minute bars used to construct the current 5-minute close bar.

Dropped returns are not exported as a separate dataset. They are recorded only for
debugging and audit purposes in `data/intermediate/preprocessing_report.json`.

The final analysis dataset will be produced after length standardization and saved
as:

```text
data/final_analysis/eurusd_5m_log_returns_final.csv
```

**Preprocessing Results**

Current preprocessing results:

```text
raw_rows_loaded: 3,671,254
raw_exact_duplicate_rows_dropped: 360
clean_1m_rows: 3,670,894
ohlc_5m_nonempty_rows: 737,034
ohlc_5m_complete_rows: 726,627
ohlc_5m_partial_rows: 10,407
return_rows_clean: 735,706
return_rows_dropped: 1,327
```

Distribution of observed 1-minute bars per retained 5-minute OHLC bar:

```text
1 observed 1-minute bar: 316
2 observed 1-minute bars: 637
3 observed 1-minute bars: 1,647
4 observed 1-minute bars: 7,807
5 observed 1-minute bars: 726,627
```
