import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("enhanced_smart_waste_dataset_v1.csv")

# ---------------- COLUMN MAPPING ----------------
df.rename(columns={
    "ItemName": "Item"
}, inplace=True)

# ---------------- FEATURE ENGINEERING ----------------
df["Produced_Qty"] = df["StockQty"]
df["Sold_Qty"] = df["DailySaleAvg"]

# Waste Quantity
df["Waste_Qty"] = df["StockQty"] * df["SpoilageChance"]

# Price & Revenue
df["Price_Per_Unit"] = np.random.randint(10, 100, size=len(df))
df["Revenue"] = df["Sold_Qty"] * df["Price_Per_Unit"]

# Waste Loss (IMPORTANT)
df["Waste_Loss"] = df["Waste_Qty"] * df["Price_Per_Unit"]

# Expiry Days
df["Expiry_Days"] = df["DaysUntilExpiry"]

# Day of Week
df["Day_of_Week"] = np.random.choice(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    size=len(df)
)

# Date
df["Date"] = pd.date_range(start="2023-01-01", periods=len(df), freq="D")

# ---------------- FINAL DATA ----------------
final_df = df[[
    "Item",
    "Category",
    "Produced_Qty",
    "Sold_Qty",
    "Waste_Qty",
    "Waste_Loss",
    "Price_Per_Unit",
    "Revenue",
    "Expiry_Days",
    "Day_of_Week",
    "Date"
]]

final_df.to_csv("retail_food_waste_data.csv", index=False)

print("Dataset converted successfully!")
