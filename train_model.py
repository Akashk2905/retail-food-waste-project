from sklearn.model_selection import GridSearchCV
import shap
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv("retail_food_waste_data.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# Encoding
le_item = LabelEncoder()
le_category = LabelEncoder()
le_day = LabelEncoder()

df["Item"] = le_item.fit_transform(df["Item"])
df["Category"] = le_category.fit_transform(df["Category"])
df["Day_of_Week"] = le_day.fit_transform(df["Day_of_Week"])

# Feature Engineering
df["Sales_Ratio"] = df["Sold_Qty"] / df["Produced_Qty"]
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

features = [
    "Item", "Category", "Day_of_Week",
    "Produced_Qty", "Sold_Qty",
    "Price_Per_Unit", "Revenue",
    "Expiry_Days", "Sales_Ratio"
]

X = df[features]

# Targets
y_qty = df["Waste_Qty"]
y_loss = df["Waste_Loss"]

# Time Split
train_size = int(len(df) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y_qty[:train_size]
y_test = y_qty[train_size:]

# Train model

print("\n Running Hyperparameter Tuning...")

params = {
    "n_estimators": [200, 300],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1]
}

grid = GridSearchCV(
    xgb.XGBRegressor(objective="reg:squarederror"),
    params,
    cv=3,
    scoring="neg_mean_absolute_error",
    verbose=1
)

grid.fit(X_train, y_train)

# Best model
model_qty = grid.best_estimator_

print("Best Parameters:", grid.best_params_)

# Prediction
pred = model_qty.predict(X_test)

print("\n Waste_Qty Model Performance")
print("MAE:", mean_absolute_error(y_test, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))

# Cross Validation
cv = cross_val_score(model_qty, X, y_qty, cv=5,
                     scoring="neg_mean_absolute_error")
print("CV MAE:", -cv.mean())

# Waste Loss model
model_loss = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05)
model_loss.fit(X_train, y_loss[:train_size])

# Save
joblib.dump(model_qty, "waste_qty_model.joblib")
joblib.dump(model_loss, "waste_loss_model.joblib")

joblib.dump(le_item, "le_item.joblib")
joblib.dump(le_category, "le_category.joblib")
joblib.dump(le_day, "le_day.joblib")

print("Models saved successfully")


explainer = shap.Explainer(model_qty)
shap_values = explainer(X)

shap.summary_plot(shap_values, X)
