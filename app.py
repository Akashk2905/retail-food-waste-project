import joblib
import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Retail Food Waste Prediction and Optimization System", layout="wide")

# =====================================================
# DATA SOURCE (UPLOAD + DEFAULT)
# =====================================================

st.sidebar.header("Data Source")

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("retail_food_waste_data.csv")

df["Date"] = pd.to_datetime(df["Date"])

# ---------------- ML MODEL LOAD ----------------
model_qty = joblib.load("waste_qty_model.joblib")
model_loss = joblib.load("waste_loss_model.joblib")

le_item = joblib.load("le_item.joblib")
le_category = joblib.load("le_category.joblib")
le_day = joblib.load("le_day.joblib")

# =====================================================
# RESTAURANT SUPPORT (Backward Compatible)
# =====================================================

if "Restaurant_Name" not in df.columns:
    df["Restaurant_Name"] = "Default_Restaurant"

restaurant = st.sidebar.selectbox(
    "Select Restaurant",
    df["Restaurant_Name"].unique()
)

df = df[df["Restaurant_Name"] == restaurant]

# =====================================================
# BASIC CALCULATIONS
# =====================================================

df["Waste_Percentage"] = (df["Waste_Qty"] / df["Produced_Qty"]) * 100
df["Waste_to_Revenue_%"] = (df["Waste_Loss"] / df["Revenue"]) * 100

rev_summary = df.groupby("Item")["Revenue"].sum().reset_index()
rev_summary["Revenue_Contribution_%"] = (
    rev_summary["Revenue"] / rev_summary["Revenue"].sum()
) * 100

df["Expiry_Risk"] = df.apply(
    lambda x: "High"
    if x["Expiry_Days"] <= 2 and x["Waste_Percentage"] > 15
    else "Normal",
    axis=1
)

# =====================================================
# ITEM SUMMARY
# =====================================================

item_summary = df.groupby("Item").agg({
    "Waste_Percentage": "mean",
    "Sold_Qty": "std"
}).reset_index()

item_summary.rename(columns={"Sold_Qty": "Demand_Variability"}, inplace=True)

item_summary = item_summary.merge(
    rev_summary[["Item", "Revenue_Contribution_%"]],
    on="Item",
    how="left"
)

scaler = MinMaxScaler()
item_summary[["Waste_Score", "Variability_Score"]] = scaler.fit_transform(
    item_summary[["Waste_Percentage", "Demand_Variability"]]
)

item_summary["Overstock_Risk_Score"] = (
    item_summary["Waste_Score"] * 0.7 +
    item_summary["Variability_Score"] * 0.3
)

item_summary["Performance_Score"] = (
    item_summary["Revenue_Contribution_%"] * 0.5 -
    item_summary["Waste_Percentage"] * 0.5
)


def categorize_risk(score):
    if score >= 0.7:
        return "High Risk"
    elif score >= 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"


def production_recommendation(waste_percent):
    if waste_percent > 20:
        return "Reduce Production by 15%"
    elif waste_percent >= 10:
        return "Reduce Production by 5%"
    else:
        return "Maintain Production Level"


item_summary["Risk_Category"] = item_summary["Overstock_Risk_Score"].apply(
    categorize_risk)
item_summary["Recommendation"] = item_summary["Waste_Percentage"].apply(
    production_recommendation)

item_summary["Avg_Production"] = df.groupby(
    "Item")["Produced_Qty"].mean().values
item_summary["Suggested_Production"] = (
    item_summary["Avg_Production"] *
    (1 - item_summary["Waste_Percentage"] / 100)
)
item_summary["Reduction_Units"] = (
    item_summary["Avg_Production"] -
    item_summary["Suggested_Production"]
)

df = df.merge(
    item_summary[["Item", "Risk_Category", "Recommendation"]],
    on="Item",
    how="left"
)

# =====================================================
# SIDEBAR FILTERS
# =====================================================

st.sidebar.header("Filters")

min_date = df["Date"].min()
max_date = df["Date"].max()

start_date, end_date = st.sidebar.date_input(
    "Select Date Range",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

category = st.sidebar.selectbox(
    "Select Category", ["All"] + list(df["Category"].unique())
)

item = st.sidebar.selectbox(
    "Select Item", ["All"] + list(df["Item"].unique())
)

view_mode = st.sidebar.radio(
    "Expiry View Mode",
    ["Operational View", "Strategic View"]
)

filtered_df = df[
    (df["Date"] >= pd.to_datetime(start_date)) &
    (df["Date"] <= pd.to_datetime(end_date))
]

if category != "All":
    filtered_df = filtered_df[filtered_df["Category"] == category]

if item != "All":
    filtered_df = filtered_df[filtered_df["Item"] == item]

    st.sidebar.markdown("---")
# ---------------- ML WASTE PREDICTION ----------------

# =====================================================
# ML WASTE PREDICTION
# =====================================================

st.sidebar.markdown("---")
st.sidebar.markdown("## Waste Prediction")

item_input = st.sidebar.selectbox(
    "Select Item",
    df["Item"].unique(),
    key="ml_item"
)

category_input = st.sidebar.selectbox(
    "Select Category",
    df["Category"].unique(),
    key="ml_category"
)

day_input = st.sidebar.selectbox(
    "Select Day",
    df["Day_of_Week"].unique(),
    key="ml_day"
)

produced_input = st.sidebar.number_input(
    "Produced Quantity",
    min_value=1,
    value=100,
    key="ml_produced"
)

sold_input = st.sidebar.number_input(
    "Sold Quantity",
    min_value=0,
    value=80,
    key="ml_sold"
)

price_input = st.sidebar.number_input(
    "Price Per Unit",
    min_value=0.0,
    value=10.0,
    key="ml_price"
)

expiry_input = st.sidebar.number_input(
    "Expiry Days",
    min_value=0,
    value=1,
    key="ml_expiry"
)

if st.sidebar.button("Predict Waste", key="ml_predict"):

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            params={
                "item": item_input,
                "category": category_input,
                "day": day_input,
                "produced_qty": produced_input,
                "sold_qty": sold_input,
                "price": price_input,
                "expiry_days": expiry_input
            }
        )

        result = response.json()

        if "error" in result:
            st.error(result["error"])
        else:
            st.markdown("##  Prediction Results")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Predicted Waste Quantity",
                          f"{result['Predicted_Waste_Qty']} units")

            with col2:
                st.metric("Predicted Waste Loss",
                          f"₹ {result['Predicted_Waste_Loss']}")

            st.caption("Prediction via API (Production Ready 🚀)")

    except Exception as e:
        st.error(f"API Connection Error: {e}")

# =====================================================
# HELPER FUNCTIONS
# =====================================================


def highlight_risk(row):
    if row["Risk_Category"] == "High Risk":
        return ["background-color: #ff4b4b"] * len(row)
    elif row["Risk_Category"] == "Medium Risk":
        return ["background-color: #ffa500"] * len(row)
    else:
        return ["background-color: #2ecc71"] * len(row)


def generate_pdf():
    file_path = "Retail_Waste_Report.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph(f"{restaurant} - Waste Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(
        f"Total Revenue: ₹{filtered_df['Revenue'].sum():,.0f}",
        styles["Normal"]
    ))
    elements.append(Paragraph(
        f"Total Waste Loss: ₹{filtered_df['Waste_Loss'].sum():,.0f}",
        styles["Normal"]
    ))

    elements.append(Spacer(1, 12))

    table_data = [["Item", "Waste %", "Risk"]]
    for _, row in item_summary.iterrows():
        table_data.append([
            row["Item"],
            f"{row['Waste_Percentage']:.2f}",
            row["Risk_Category"]
        ])

    elements.append(Table(table_data))
    doc.build(elements)
    return file_path

# =====================================================
# DASHBOARD
# =====================================================


st.title("Retail Food Waste Prediction and Optimization System")
st.subheader(f"Restaurant: {restaurant}")

col1, col2, col3 = st.columns(3)

col1.metric("Total Revenue (₹)", f"{filtered_df['Revenue'].sum():,.0f}")
col2.metric("Total Waste Loss (₹)", f"{filtered_df['Waste_Loss'].sum():,.0f}")
col3.metric("Total Waste Units", f"{filtered_df['Waste_Qty'].sum():,.0f}")

waste_percent_of_revenue = (
    filtered_df["Waste_Loss"].sum() /
    filtered_df["Revenue"].sum()
) * 100

st.metric("Waste as Percentage of Revenue", f"{waste_percent_of_revenue:.2f}%")

# Alerts
st.markdown("---")
st.subheader("Real-Time Risk Alerts")

critical_items = filtered_df[
    (filtered_df["Waste_Percentage"] > 20) |
    (filtered_df["Expiry_Risk"] == "High")
]

if not critical_items.empty:
    st.error(
        f"{critical_items['Item'].nunique()} Critical Risk Items Detected")
else:
    st.success("No critical alerts")

# Heatmap
st.markdown("---")
st.subheader("Waste Heatmap")

pivot = filtered_df.pivot_table(
    values="Waste_Percentage",
    index="Item",
    columns="Day_of_Week",
    aggfunc="mean"
)

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(pivot, cmap="Reds", ax=ax)
st.pyplot(fig)

# Expiry Analysis
st.markdown("---")
st.subheader("Expiry Risk Analysis")

if view_mode == "Operational View":
    operational_df = filtered_df[
        filtered_df["Expiry_Risk"] == "High"
    ][["Date", "Item", "Expiry_Days", "Waste_Percentage"]]
    st.dataframe(operational_df)
else:
    strategic_df = filtered_df[
        filtered_df["Expiry_Risk"] == "High"
    ].groupby("Item").agg({
        "Expiry_Days": "mean",
        "Waste_Percentage": "mean"
    }).reset_index()
    st.dataframe(strategic_df)

# Production Engine
st.markdown("---")
st.subheader("Production Recommendation Engine")

styled_table = item_summary[
    [
        "Item", "Avg_Production", "Suggested_Production",
        "Reduction_Units", "Waste_Percentage",
        "Risk_Category", "Recommendation"
    ]
].sort_values("Waste_Percentage", ascending=False)

st.dataframe(
    styled_table.style.apply(highlight_risk, axis=1),
    use_container_width=True
)

# Performance Ranking
st.markdown("---")
st.subheader("Item Performance Ranking")

st.dataframe(
    item_summary[
        [
            "Item", "Revenue_Contribution_%", "Waste_Percentage",
            "Overstock_Risk_Score", "Performance_Score"
        ]
    ].sort_values("Performance_Score", ascending=False)
)

# Scenario Simulation
st.markdown("---")
st.subheader("Scenario Simulation")

reduction_slider = st.slider("Select Waste Reduction %", 5, 30, 15)
potential_savings = filtered_df["Waste_Loss"].sum() * (reduction_slider / 100)

st.metric("Potential Savings (₹)", f"{potential_savings:,.0f}")

# Executive Summary
st.markdown("---")
st.subheader("Executive Summary")

high_risk_count = item_summary[item_summary["Risk_Category"]
                               == "High Risk"].shape[0]

st.write(f"""
Total Waste Loss: ₹{filtered_df['Waste_Loss'].sum():,.0f}  
Waste Percentage of Revenue: {waste_percent_of_revenue:.2f}%  
High Risk Items: {high_risk_count}  
Suggested Waste Reduction: {reduction_slider}%  
Estimated Savings: ₹{potential_savings:,.0f}
""")

# PDF Download
st.markdown("---")
st.subheader("Download Report")

if st.button("Generate PDF Report"):
    pdf_path = generate_pdf()
    with open(pdf_path, "rb") as f:
        st.download_button(
            "Download Report",
            f,
            file_name="Retail_Waste_Report.pdf"
        )
