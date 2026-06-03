# ════════════════════════════════════════════════════════════════════
# data_pipeline.py
# Generates synthetic retail data with known ground truth for a 
# causal inference study on the impact of a pricing pilot.
# Author: Sabiha Farhat
# ════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import holidays as hol

# Set random seed so this script produces identical data every time
# it runs — essential for reproducibility of the analysis
np.random.seed(42)


# ─── TABLE 1: STORES ────────────────────────────────────────────────
# One row per store. Captures static attributes that don't change 
# over time. Used as the join key for all downstream tables.
# 
# Treatment design:
#   Stores 1–8  → physical pilot stores (receive price change)
#   Store 9     → online channel as virtual store (placeholder — 
#                 final treatment status pending industry expert input)
#   Stores 10–20 → physical control stores (no price change)
# ────────────────────────────────────────────────────────────────────

# Store IDs: integers 1 through 20
store_id = list(range(1, 21))

# Store names: real-ish Waterloo region neighbourhood labels for 
# authenticity; readable in charts and reports
store_name = [
    "Kitchener-Fairview", "Kitchener-Stanley Park", "Kitchener-Downtown",
    "Waterloo-Uptown", "Waterloo-University", "Waterloo-Conestoga",
    "Cambridge-Hespeler", "Cambridge-Galt",
    "Online-Waterloo",
    "Kitchener-Forest Hill", "Kitchener-Doon", "Kitchener-Chicopee",
    "Waterloo-Lakeshore", "Waterloo-Beechwood", "Waterloo-Columbia",
    "Cambridge-Preston", "Cambridge-Blair", "Cambridge-North",
    "Kitchener-Activa", "Waterloo-Erbsville"
]

# City: one of the three Waterloo-region cities, or "Online" for 
# the virtual store
city = [
    "Kitchener", "Kitchener", "Kitchener",
    "Waterloo", "Waterloo", "Waterloo",
    "Cambridge", "Cambridge",
    "Online",
    "Kitchener", "Kitchener", "Kitchener",
    "Waterloo", "Waterloo", "Waterloo",
    "Cambridge", "Cambridge", "Cambridge",
    "Kitchener", "Waterloo"
]

# Store size in square feet — used downstream for store similarity matching and as the basis for store_format classification.
# Realistic range for mid-format retail: 15,000 to 80,000 sq ft
store_size_sqft = np.random.randint(15000, 80001, size=20)

# Average weekly customer footfall in the pre-intervention period.
# Used downstream for store similarity matching.
weekly_footfall = np.random.randint(5000, 21000, size=20)

# Treatment flag: True for the 8 physical pilot stores AND store 9 (online, placeholder). Final online treatment status pending 
# expert input — can be flipped with one line change.
is_pilot = [True if sid <= 9 else False for sid in store_id]

# Store format classification derived from size in square feet:
#   Small  → under 30,000 sq ft  
#   Medium → 30,000 to 49,999 sq ft  
#   Large  → 50,000 sq ft and above
# Used in HTE analysis (Notebook 06) to assess whether store size 
# moderates the treatment effect.
store_format = [
    'small' if size < 30000
    else ('medium' if size < 50000 else 'large')
    for size in store_size_sqft
]

# Placeholder for the matched control store ID. Populated later in 01_data_prep.ipynb after running store similarity matching.
matched_control_id = [None] * 20

# Store opening date. All stores set to 2023-01-01 so every store satisfies the "12+ months open before pilot" eligibility filter 
# (pilot launches Q2 2024).
open_since = ["2023-01-01"] * 20

# Combine all columns into a single pandas DataFrame
stores_df = pd.DataFrame({
    "store_id": store_id,
    "store_name": store_name,
    "city": city,
    "store_size_sqft": store_size_sqft,
    "weekly_footfall": weekly_footfall,
    "is_pilot": is_pilot,
    "store_format": store_format,
    "matched_control_id": matched_control_id,
    "open_since": open_since,
})

# Save to CSV in the data/ folder. index=False removes the auto-
# generated row index so the CSV stays clean.
stores_df.to_csv('data/stores.csv', index=False)

print(f"✅ Stores table: {len(stores_df)} rows saved to data/stores.csv")
# print(stores_df)

# -----------------------------------------------------------------------------------------------------------------

# Department table by stores
dept_ids_list = [1, 2, 3, 4, 5, 6]

dept_id = dept_ids_list * 20

dept_name_map = {
    1: "Electronics",
    2: "Home & Kitchen",
    3: "Sports & Outdoors",
    4: "Clothing",
    5: "Books",
    6: "Toys"
}

dept_name = [dept_name_map[d] for d in dept_id]

dept_pilot_map = {
    # dept_id 1, 2, 3 received the price change
    1: True, 2: True, 3: True,
    4: False, 5: False, 6: False
}

dept_pilot = [dept_pilot_map[d] for d in dept_id]

store_id_dept = np.repeat(store_id,6)

dept_elasticity_map = {
    1: "Low",      # Electronics
    2: "Medium",   # Home & Kitchen
    3: "High",     # Sports & Outdoors
    4: "Medium",   # Clothing
    5: "Medium",   # Books
    6: "Medium",   # Toys
}

price_elasticity_group = [dept_elasticity_map[d] for d in dept_id]

dept_df = pd.DataFrame(
    {
      "dept_id": dept_id,
      "dept_name": dept_name,
      "dept_pilot": dept_pilot,
      "store_id_dept": store_id_dept,
      "price_elasticity_group": price_elasticity_group
    }
)
# print(dept_df)
dept_df.to_csv('data/dept.csv', index=False)

"""
─── TABLE 3: DAILY SALES (Session 1 — structural foundation) ─────────
One row per store × department × date.
Timeline: Apr 2023 → Sep 2024 (18 months)
- Pre-pilot:  Apr 2023 – Mar 2024 (12 months baseline)
- Pilot:      Apr 2024 – Jun 2024 (3 months — treatment active)
- Post-pilot: Jul 2024 – Sep 2024 (3 months — observation only)
Session 1 builds the grid + structural columns (no revenue yet).
──────────────────────────────────────────────────────────────────────

"""

PILOT_START = pd.Timestamp("2024-04-01")
PILOT_END = pd.Timestamp("2024-06-30")
START_DATE = "2023-04-01"
END_DATE = "2024-09-30"

# Create date range 
dates = pd.date_range(start=START_DATE, end=END_DATE, freq="D")

# Ontario public holidays for 2023 and 2024
ontario_holidays = hol.Canada(prov='ON', years=[2023, 2024])

# Append dates, dept_ids, and store_ids into one list to later transform the list into dataframe

all_rows = []
for s in store_id:
    for d in dept_ids_list:
        for date in dates:
            all_rows.append({'store_id': s, 'dept_id': d, 'dates':date})

daily_sales_df = pd.DataFrame(all_rows)
# print(daily_sales_df)

# Step C — Add derived columns

# week_start_date: the Monday of the week
daily_sales_df['week_start_date'] = daily_sales_df['dates'] - pd.to_timedelta(daily_sales_df['dates'].dt.weekday, unit='D')

# during_pilot: True only during Apr–Jun 2024
daily_sales_df['during_pilot'] = (daily_sales_df['dates'] >= PILOT_START) & (daily_sales_df['dates'] <= PILOT_END)

daily_sales_df['post_pilot'] = daily_sales_df['dates'] > PILOT_END

daily_sales_df['is_store_open'] = [d not in ontario_holidays for d in daily_sales_df['dates']]

daily_sales_df['reason_closed'] = np.where(daily_sales_df['is_store_open'], None, "public_holiday")

# Create pilot stores and dept set as a lookup reference to flag them in daily_sales_df
pilot_stores = set(stores_df.loc[stores_df['is_pilot'] == True, 'store_id'])

pilot_dept = set(dept_df.loc[dept_df['dept_pilot'] == True, 'dept_id'])

daily_sales_df['treated'] = np.where((daily_sales_df['store_id'].isin(pilot_stores)) & (daily_sales_df['dept_id'].isin(pilot_dept)) & (daily_sales_df['during_pilot'] == True), True, False)



# Revenue Column
# Bring store_size_sqft into daily_sales_df temporarily

daily_sales_df = daily_sales_df.merge(stores_df[['store_id', 'store_size_sqft']] , on = 'store_id', how='left')

# Calculate size_factor: how big is this store relative to average?
mean_size = stores_df['store_size_sqft'].mean()
daily_sales_df['size_factor'] = daily_sales_df['store_size_sqft'] / mean_size

dept_revenue_mid = {1: 55000, 2: 37500, 3: 25000,
                    4: 32500, 5: 10000, 6: 14000}

# Map dept midpoint to each row, divide by 7 for daily
# daily_sales_df['dept_mid_daily'] = [dept_revenue_mid[d]/7 for d in daily_sales_df['dept_id'] ]
daily_sales_df['dept_mid_daily'] = (daily_sales_df['dept_id'].map(dept_revenue_mid)/7)

daily_sales_df['base_revenue'] = (
    daily_sales_df['dept_mid_daily'] * daily_sales_df['size_factor']
)

# Inject seasonality
def quater_factor(month):
    if month in [10, 11, 12]:
        return 1.3
    elif month in [1,2,3]:
        return 0.85
    else:
        return 1.00
    
# Day of week factor (5=Sat, 6=Sun)
def day_factor(dayofweek):
    return 1.20 if dayofweek >= 5 else 1.00

# Apply both
daily_sales_df['seasonality'] = (
    daily_sales_df['dates'].dt.month.map(quater_factor) * daily_sales_df['dates'].dt.dayofweek.map(day_factor)
)

noise = np.random.normal(loc=1.0, scale=0.05, size=len(daily_sales_df)) 

daily_sales_df['revenue'] = (
    daily_sales_df['base_revenue'] *
    daily_sales_df['seasonality'] *
    noise
)

true_att = {1: 0.12, 2: 0.03, 3: -0.02}

# For each treated row, multiply revenue by (1 + true_att)
# For non-treated rows, revenue unchanged
daily_sales_df['revenue'] = np.where(
    daily_sales_df['treated'],
    daily_sales_df['revenue'] * (
        1 + daily_sales_df['dept_id'].map(true_att).fillna(0)
    ),
    daily_sales_df['revenue']
    )
daily_sales_df['revenue'] = np.where(
    daily_sales_df['is_store_open'] == True, daily_sales_df['revenue'] , 0)


# Approximate units per dollar of revenue — dept-level basket size
# This is a simulation parameter, not a real price claim
dept_units_per_dollar = {
    1: 0.007,   # Electronics — expensive items, fewer units
    2: 0.022,   # Home & Kitchen — mid-range
    3: 0.029,   # Sports & Outdoors — mid-range
    4: 0.033,   # Clothing — lower price point
    5: 0.067,   # Books — cheap, many units
    6: 0.050    # Toys — cheap-mid
}

daily_sales_df['units_sold'] = (
    daily_sales_df["revenue"]*daily_sales_df['dept_id'].map(dept_units_per_dollar)
).astype(int)

daily_sales_df = daily_sales_df.drop(
    columns=[
        'store_size_sqft', 'size_factor', 
             'dept_mid_daily', 'base_revenue', 'seasonality'
    ]
)
# print(daily_sales_df.head(5))
daily_sales_df.to_csv('data/daily_sales.csv', index=False)


ground_truth = pd.DataFrame(
    {
        'dept_id': [1,2,3],
        'dept_name': ['Electronics', 'Home & Kitchen', 'Sports & Outdoors'],
        'true_att':  [0.12, 0.03, -0.02],
        'elasticity_group': ['Low', 'Medium', 'High'],
        'treatment_start': ['2024-04-01'] * 3,
        'treatment_end':   ['2024-06-30'] * 3,
    }
)

print(ground_truth)
