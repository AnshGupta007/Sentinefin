# Pre-Registered Backtest Cases (Phase 5)

These cases and success criteria were written **before** any backtest run, per
the master prompt's anti-leakage protocol.

## Success criterion (fixed in advance)

> A case counts as a success if SentinelFin flags (alert = reconstruction error
> above the configured threshold) a cluster whose example narratives describe
> the case's harm pattern at least `lead_time_months_required` month(s) before
> the official-recognition date below. Lead time is reported honestly whether
> positive or negative. Models are retrained from scratch on data truncated to
> end before the recognition date; no trained state from the full run is reused.

## Case 1: Buy Now, Pay Later (BNPL)

- Product: "Buy Now, Pay Later (BNPL)" complaints
- First appearance in CFPB data: BNPL-related narratives appear sporadically
  from roughly 2019 onward under debt-collection/credit-reporting products.
- Official recognition date: **2022-03-01**
- Basis: CFPB opened its BNPL market inquiry in December 2021 and published its
  first BNPL market report in September 2022
  (https://www.consumerfinance.gov/about-us/newsroom/cfpb-launches-inquiry-into-buy-now-pay-later-credit/).
- Sources: CFPB newsroom; CFPB BNPL market report (Sep 2022).

## Case 2: Earned Wage Access (EWA)

- Product: "Earned Wage Access" complaints
- Official recognition date: **2023-06-01**
- Basis: CFPB began product-specific EWA scrutiny and guidance activity through
  2023 (Circular 2022-06 on EWA fee disclosure; subsequent 2023–2024 EWA
  interpretive rulemaking).
- Sources: CFPB Compliance Circular 2022-06; CFPB EWA interpretive rule (2024).

## False-alarm estimation

Across all alerts raised on the full historical run, the report estimates the
fraction whose top terms do not correspond to either pre-registered case or to
another recognizable real pattern. This is an approximate, manually-reviewable
heuristic and is labeled as such in the output.
