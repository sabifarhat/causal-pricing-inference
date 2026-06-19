# Methodology Notes — Causal Pricing Inference Project

Detailed explanations, design decisions, and learning notes from the project.
These notes supplement the concise insights in each notebook with deeper context
for walkthrough and interview preparation.

**Author:** Sabiha Farhat
**Repo:** github.com/sabifarhat/causal-pricing-inference

---

## 1. Study Design — Why Store-Level Analysis

### Unit of analysis
This project measures the impact of a pricing tool rollout at **store level** —
did the 8 pilot stores outperform the 11 control stores? The unit of treatment
is the store (the tool was deployed to entire stores, not to individual products),
so the unit of analysis must also be the store.

### Why not SKU-level?
In production retail analytics (Staples, Amazon, Walmart), pricing impact is
typically measured at **SKU level** — comparing repriced products vs non-repriced
products within the SAME store. This eliminates store-level noise entirely because
you are comparing products on the same shelf, in the same store, with the same
customers.

This project does not include SKU-level data because:
- Generating realistic SKU-level pricing data is enormously complex (hundreds of
  SKUs per department, each with different elasticities and substitution effects)
- The causal methods demonstrated (DiD, Synthetic Control, BSTS, HTE) all work
  at store or store x department level — their standard application
- A price events table mapping individual SKU price changes would be required

**For walkthrough:** "This project measures the impact of the pricing tool at store
level. A production extension would add SKU-level within-store controls, which
requires a price events table — intentionally excluded to keep scope focused on
causal inference methods."

---

## 2. Store Matching — Why Post-Hoc and Not Pre-Assignment

### The question
Since this is synthetic data, we could have built matched pairs in data_pipeline.py
and assigned treatment based on those pairs. Doing matching after the fact on
synthetic data where WE chose which stores to treat looks like a post-hoc fix
for a flaw we introduced ourselves.

### The answer
The project intentionally simulates a **real-world observational study** where
stores were pre-selected by a business decision, not randomised. This is realistic
for retail — businesses don't randomly assign stores to pricing pilots, they choose
stores for operational reasons (willing store manager, regional focus, logistics).

Matching is applied post-hoc to reduce selection bias — exactly what practitioners
do every day at companies like Staples, Loblaw, and Walmart Canada.

### When matching is used in real life
- **Retail:** Store-level pricing or promotion rollouts where pilot stores are
  pre-selected by management
- **Tech:** Feature rollout to a subset of users in one city before national launch
- **Healthcare:** New treatment protocol given to patients in one ward, matched to
  similar patients in other wards
- **Policy:** Minimum wage increase in one province, economists match counties
  across provinces (Card & Krueger study, 2021 Nobel Prize in Economics)

### Matching method chosen
Euclidean distance on 3 z-scored features: store_size_sqft, weekly_footfall,
pre_period_avg_weekly_rev. Greedy 1:1 matching — closest control store wins,
no repeat matches.

Alternative considered: **Hungarian Algorithm** (scipy.optimize.linear_sum_assignment)
for globally optimal matching. Not used because with only 8 pilot stores and 11
controls, the difference between greedy and optimal is negligible.

### Propensity scores
Not applicable — propensity score matching requires fitting a logistic regression
to predict treatment assignment. With only 20 stores, there are too few observations
to fit a meaningful model. Propensity scoring is standard in industry with hundreds
or thousands of units (users, customers, stores).

---

## 3. A/A Test — Why It Failed and What It Means

### Result
p-value = 0.0016 — Group A (CAD 175,182) and Group B (CAD 194,942) showed a
statistically significant difference despite both being control stores.

### Root cause
Small sample size (n=11 control stores). Random 5/6 split by chance assigned
larger stores to Group B and smaller stores to Group A. With only 11 stores,
random halving cannot guarantee size-balanced groups.

### Implication
This is a sample size limitation, not a data quality issue. It directly motivates
using matched pairs instead of random store assignment for the A/B test.

### In production
With hundreds of stores, propensity score matching via logistic regression would
be used. At n=20 stores, Euclidean distance matching on z-scored features is the
appropriate method.

---

## 4. Power Analysis — Why MDE is 56.1%

### What MDE means
Minimum Detectable Effect — the smallest revenue lift the t-test can reliably
detect at 80% power and 5% significance. With n=8 pilot stores and n=11 control
stores, the MDE is 56.1%.

### What this means in practice
The test needs to see a 56% revenue lift to call it statistically significant.
The true treatment effects (12%, 3%, -2%) are far below this threshold. The
insignificant A/B test results were mathematically expected before the test was
even run.

### Industry threshold
5% MDE is the standard for mid-format retail. Our 56.1% is 11x higher than the
industry benchmark — severely underpowered.

---

## 5. Understanding Standard Error, Variance, and Why CAD 6K is Undetectable

### Standard deviation
Measures how spread out individual store revenues are around the group average.
Pilot store weekly revenues range from roughly CAD 15,000 to CAD 90,000 — a
large spread driven by natural differences in store size, location, and customer
base, not by the treatment.

### Standard error
Measures how reliable the group average is as an estimate.
Formula: Standard Error = Standard Deviation / sqrt(sample size).
With only 8 pilot stores and high standard deviation, the standard error is
large — the pilot group average has a wide margin of uncertainty.

### Why CAD 6,000 is undetectable despite being real
The true Electronics treatment effect is +12%. For a store averaging CAD 50,000/week,
that is roughly CAD 6,000/week. This CAD 6,000 difference genuinely exists in the data.

However, with high store-to-store variance, random chance alone could produce
differences of CAD 5,000–15,000 between any two groups of 8 stores — even without
any treatment. The t-test asks: "Could this CAD 6,000 difference have appeared by
chance?" With this much variance and only 8 stores, the answer is "yes, easily."
So it returns p > 0.05.

If variance were low (all stores doing roughly the same revenue), a CAD 6,000
difference would stand out immediately — there would be no plausible way random
chance could produce it.

**Key insight:** Statistical significance is not about whether a difference exists.
It is about whether the difference is large enough relative to the noise that it
could not have happened by chance.

### The t-test formula in plain English
t-statistic = (Pilot mean - Control mean) / Standard Error of the difference

If numerator (the actual difference) is large relative to denominator (the noise),
t-statistic is large, p-value is small, result is significant.
If noise overwhelms the signal, t-statistic is small, p-value is large, result
is insignificant.

---

## 6. A/B Test Results — Directionally Correct but Insignificant

### Results using matched pairs

| Department | Lift | p-value | Significant |
|---|---|---|---|
| Electronics | +9.32% | 0.1171 | No |
| Home & Kitchen | +0.77% | 0.8910 | No |
| Sports & Outdoors | -3.84% | 0.4852 | No |

### Why "directionally correct"
Ground truth: Electronics +12%, H&K +3%, Sports -2%.
A/B test: Electronics positive, H&K positive, Sports negative.
The signs match — the test is pointing the right way but is not statistically
confident enough to claim it due to small sample size.

### How to improve A/B test results in production
1. **More stores** — n=40+ per group reduces MDE to detectable levels
2. **Longer pilot** — 6-12 months instead of 13 weeks reduces variance
3. **CUPED** — use pre-period revenue as covariate, reduces variance by 30-50%
   without adding stores. Industry standard at Netflix, Booking, Microsoft.
4. **Paired t-test** — compare each pilot store to its matched control directly
5. **One-tailed test** — if directional hypothesis exists, reduces MDE by ~20%
6. **SKU-level data** — within-store controls eliminate store-level noise entirely

### On unit of analysis — why not store x department
Using store x department combinations would triple the sample size (24 pilot vs
33 control observations). However, department revenues within the same store are
correlated — if footfall drops at one store, ALL departments drop together. They
share the same customers, store size, and management.

Feeding correlated observations into a t-test underestimates variance and
produces artificially low p-values (inflated Type 1 error / false positives).
The fix would be a mixed effects model with store-level random effects or
clustered standard errors — beyond the scope of a simple t-test.

### Alternative statistical tests considered
- **Mann-Whitney U** — non-parametric, does not assume normality
- **Permutation test** — assumption-free, suitable for n=8 small samples
- **Mixed effects model** — proper fix for store x department correlated data
- **Clustered standard errors** — OLS with variance clustered at store level

T-test was chosen for interpretability and consistency with industry standard
A/B test reporting at store level.

---

## 7. Difference-in-Differences — Key Concepts

### What DiD solves that A/B test cannot
The A/B test compares pilot vs control ONLY during the pilot period. It cannot
distinguish between:
- Revenue lift caused by the price change
- Revenue lift that would have happened anyway (seasonal trends, market growth)
- Pre-existing differences between pilot and control stores

DiD fixes this by comparing the CHANGE in pilot stores to the CHANGE in control
stores over the same time period. The control group acts as a "what would have
happened anyway" benchmark.

### The 2x2 Table — Core Intuition

| | Pre-pilot | During pilot | Change |
|---|---|---|---|
| Control stores | CAD 58,066 | CAD 57,622 | -444 |
| Pilot stores | CAD 55,684 | CAD 61,755 | +6,071 |

```
DiD = Pilot change − Control change
    = +6,071 − (−444)
    = CAD 6,515
```

Control stores dropped CAD 444 naturally — no treatment applied.
Pilot stores gained CAD 6,071 in total.
Subtracting the natural drift isolates the true causal effect: CAD 6,515/week.

### What each OLS regression coefficient means
The DiD regression model: `revenue = b0 + b1(treated) + b2(post) + b3(did)`

- **Intercept (b0 = 58,066):** Control stores, pre-pilot baseline — the reference point
- **treated (b1 = -2,383):** Pre-existing gap — pilot stores were already CAD 2,383
  lower than control stores BEFORE the pilot started. Unrelated to treatment.
- **post (b2 = -444):** Natural time trend — control stores dropped CAD 444 between
  periods with NO treatment. Affects all stores regardless of treatment.
- **did (b3 = 6,515):** The causal effect — after removing the pre-existing gap and
  natural trend, pilot stores gained CAD 6,515/week due to the price change.

### Why DiD gives a larger estimate than the naive A/B test
- Naive A/B difference: CAD 5,265 (p=0.1171)
- DiD estimate: CAD 6,515 (p=0.0694)

The A/B test was UNDERSTATING the effect. Pilot stores started from a lower
baseline (CAD 2,383 gap). DiD accounts for this pre-existing gap and reveals
the true causal effect is larger than the simple snapshot comparison showed.

### Parallel trends assumption
DiD only works if, without treatment, pilot and control stores would have followed
the same revenue trend. This is validated two ways:
1. **Visually** — Chart 2 in Notebook 01 shows lines moving together pre-pilot
2. **Formally** — t-test on week-over-week growth rates in the pre-period.
   p > 0.05 confirms no statistically significant difference in growth rates.

---

## 8. P-value Interpretation — When You Want High vs Low

The p-value always means the same thing: "probability this difference happened
by random chance." But whether you WANT a high or low p-value depends on the test.

| Test | Desired p-value | Reason |
|---|---|---|
| A/A test | p > 0.05 | Want groups to be similar — confirms no spurious differences |
| Parallel trends test | p > 0.05 | Want groups to trend similarly — validates DiD assumption |
| A/B test | p < 0.05 | Want to confirm treatment caused a real effect |
| DiD 'did' coefficient | p < 0.05 | Want to confirm treatment caused a real effect |

**Key insight:** p < 0.05 means "this difference is too large to have happened by
random chance." Whether that's good or bad depends entirely on what you're testing.
For validation tests (A/A, parallel trends) you want no significant difference.
For effect tests (A/B, DiD) you want a significant difference.

---

## 9. CUPED vs BSTS — Why They Are Different

### CUPED (Controlled experiment Using Pre-Experiment Data)
- Uses pre-period data as a **covariate** to reduce variance in the outcome
- Still a t-test at its core — just on an adjusted outcome
- Goal: **reduce noise** so you can detect smaller effects
- No time series modelling, no forecasting
- Simple: ~10 lines of code

### BSTS (Bayesian Structural Time Series)
- Builds a **full time series model** of what revenue would have looked like
  without treatment
- Uses control stores' entire weekly time series to construct a counterfactual
- Goal: **estimate the causal effect** by comparing actual vs predicted
- Models trends, seasonality, and structural breaks over time
- Complex: full Bayesian model

### Key difference
CUPED asks: "How do I reduce noise in my t-test using pre-period data?"
BSTS asks: "What would revenue have looked like if the pilot never happened?"

Covariate (CUPED) = observed data used to adjust for noise.
Counterfactual (BSTS) = estimated alternate reality that never happened.

---

## 11. DiD — Layman Explanation and Key Intuition

### How to explain DiD to anyone in plain English

"Imagine you change prices in 8 stores in April. By June your revenue went up
by CAD 6,071. But maybe revenue would have gone up anyway — busy season,
customers spending more everywhere.

So you look at 11 similar stores that did NOT change prices. Their revenue
dropped by CAD 444 over the same period.

Now you can separate what's yours from what's the market:
Your stores gained 6,071. But the market says things should have dropped 444.
So your price change actually caused a gain of 6,071 + 444 = CAD 6,515.

That extra CAD 6,515 — the part that can't be explained by market trends —
that's the true impact of your pricing decision. That's DiD."

### The swimming upstream analogy
The market was pulling revenue DOWN by 444. Your stores fought against that
pull AND still came out ahead by 6,071. Your price change had to:
1. Overcome the natural -444 drop
2. AND deliver the +6,071 gain
True effect = 6,071 + 444 = 6,515

### What if the market went UP instead?
If control stores gained +444, some of your 6,071 was just free market
tailwind — not your pricing decision.
DiD = 6,071 - 444 = 5,627 (only this much was truly caused by price change)

### The rule — always the same formula
DiD = Pilot change - Control change

- Control went DOWN (-444) → subtracting a negative → ADD → effect is larger
- Control went UP (+444)   → subtracting a positive → SUBTRACT → effect is smaller

The formula never changes. Direction of market movement determines whether
your true effect is larger or smaller than the raw observed change.

### Why 'treated' coefficient reflects only the pre-period gap
When treated=1 and post=0 → pilot stores in PRE-period only.
When treated=1 AND post=1 → that row belongs to 'did', not 'treated'.
So OLS naturally isolates 'treated' as the structural gap in the pre-period.

The four regression combinations map perfectly to the 2x2 DiD table:
- treated=0, post=0 → Control, pre-period    → Intercept
- treated=1, post=0 → Pilot, pre-period      → Intercept + treated
- treated=0, post=1 → Control, post-period   → Intercept + post
- treated=1, post=1 → Pilot, post-period     → Intercept + treated + post + did

### Connection to regression — why 'did' = DiD formula
DiD = (Pilot change) - (Control change)
    = (post + did) - (post)
    = did

The 'post' terms cancel out automatically — the regression implements the
DiD formula exactly. Manual calculation and OLS regression give identical answers.

---

## 10. Causal Inference Methods — Learning Map

Topics covered in this project:
- SUTVA — applied (Store 9 exclusion)
- Parallel trends — visual (Chart 2) and formal test (Notebook 03)
- Store matching — Notebook 01
- A/B test with power analysis — Notebook 02
- Difference-in-Differences — Notebook 03
- Synthetic Control — Notebook 04
- BSTS / CausalImpact — Notebook 05
- HTE Meta-learners — Notebook 06

Topics not in project but important for interviews:
- **Propensity scores / IPW** — not applicable at n=20 stores, standard for
  large observational studies
- **CUPED** — mentioned as improvement, not implemented (insufficient impact at n=8)
- **Regression Discontinuity Design** — different study design, cutoff-based
- **Instrumental Variables** — for unobserved confounders
- **Power analysis (prospective)** — done retrospectively here; in real life
  done BEFORE the experiment to determine required sample size
