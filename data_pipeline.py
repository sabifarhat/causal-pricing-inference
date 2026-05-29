# ════════════════════════════════════════════════════════════════════
# data_pipeline.py
# Generates synthetic retail data with known ground truth for a 
# causal inference study on the impact of a pricing pilot.
# Author: Sabiha Farhat
# ════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np

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
print(stores_df)