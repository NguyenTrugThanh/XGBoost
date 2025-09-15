from flask import Flask, render_template, request
import joblib
import pandas as pd

# Load model + preprocessor
model = joblib.load("output_model/xgb_calibrated_model.joblib")
preprocessor = joblib.load("output_model/preprocessor.joblib")
feature_names = pd.read_csv("output_model/feature_names.csv").squeeze().tolist()

# Threshold (có thể chỉnh tùy ý, ví dụ 0.25 = 25%)
THRESHOLD = 0.25

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    label = None
    if request.method == "POST":
        # Lấy dữ liệu từ form (10 feature gốc)
        revol_util = float(request.form.get("RevolvingUtilizationOfUnsecuredLines"))
        age = float(request.form.get("age"))
        late_30_59 = float(request.form.get("NumberOfTime30-59DaysPastDueNotWorse"))
        debt = float(request.form.get("DebtRatio"))
        income = float(request.form.get("MonthlyIncome"))
        open_credit = float(request.form.get("NumberOfOpenCreditLinesAndLoans"))
        late_90 = float(request.form.get("NumberOfTimes90DaysLate"))
        real_estate = float(request.form.get("NumberRealEstateLoansOrLines"))
        late_60_89 = float(request.form.get("NumberOfTime60-89DaysPastDueNotWorse"))
        dependents = float(request.form.get("NumberOfDependents"))

        # Feature engineering
        data = {
            "RevolvingUtilizationOfUnsecuredLines": revol_util,
            "age": age,
            "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
            "DebtRatio": debt,
            "MonthlyIncome": income,
            "NumberOfOpenCreditLinesAndLoans": open_credit,
            "NumberOfTimes90DaysLate": late_90,
            "NumberRealEstateLoansOrLines": real_estate,
            "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
            "NumberOfDependents": dependents,
            # Feature engineering thêm
            "MonthlyIncome_missing": 1 if pd.isna(income) else 0,
            "NumberOfDependents_missing": 1 if pd.isna(dependents) else 0,
            "Income_per_person": income / (dependents + 1),
            "Debt_to_income_ratio": debt * (income if not pd.isna(income) else 0)
        }

        # Fill các cột khác = 0 nếu thiếu
        for col in feature_names:
            if col not in data:
                data[col] = 0

        # Chuẩn thứ tự cột
        df_new = pd.DataFrame([data])[feature_names]

        # Preprocess & predict
        X_new = preprocessor.transform(df_new)
        proba = model.predict_proba(X_new)[0, 1]
        prediction = round(proba * 100, 2)

        # Phân loại theo threshold mới
        if proba >= THRESHOLD:
            label = "Có nguy cơ vỡ nợ"
        else:
            label = "An toàn"

    return render_template("index.html", prediction=prediction, label=label, threshold=int(THRESHOLD*100))

if __name__ == "__main__":
    app.run(debug=True)
