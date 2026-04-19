from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# ✅ Load models
model_qty = joblib.load("waste_qty_model.joblib")
model_loss = joblib.load("waste_loss_model.joblib")

# ✅ Load encoders
le_item = joblib.load("le_item.joblib")
le_category = joblib.load("le_category.joblib")
le_day = joblib.load("le_day.joblib")


@app.get("/")
def home():
    return {"message": "Retail Food Waste Prediction API is running 🚀"}


# ✅ Prediction Route
@app.post("/predict")
def predict(
    item: str,
    category: str,
    day: str,
    produced_qty: float,
    sold_qty: float,
    price: float,
    expiry_days: int
):
    try:
        # 🔥 Clean input (handles lowercase issues)
        item = item.strip().title()
        category = category.strip().title()
        day = day.strip().title()

        # 🔥 Safe encoding (handles unseen values)
        def safe_encode(value, encoder):
            if value in encoder.classes_:
                return encoder.transform([value])[0]
            else:
                return 0  # fallback (no crash)

        item_enc = safe_encode(item, le_item)
        category_enc = safe_encode(category, le_category)
        day_enc = safe_encode(day, le_day)

        # ✅ Feature Engineering
        sales_ratio = sold_qty / produced_qty if produced_qty > 0 else 0
        revenue = sold_qty * price

        # ✅ Feature array
        features = np.array([[
            item_enc,
            category_enc,
            day_enc,
            produced_qty,
            sold_qty,
            price,
            revenue,
            expiry_days,
            sales_ratio
        ]])

        # ✅ Predictions
        waste_qty = model_qty.predict(features)[0]
        waste_loss = model_loss.predict(features)[0]

        return {
            "Predicted_Waste_Qty": round(float(waste_qty), 2),
            "Predicted_Waste_Loss": round(float(waste_loss), 2)
        }

    except Exception as e:
        return {"error": str(e)}
