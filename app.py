from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("linear_regression_model.pkl")

@app.post("/predict")
def predict(
    age: float,
    bmi: float,
    children: int,
    is_female: bool,
    is_smoker: bool,
    region_southeast: bool,
    bmi_category_Obese: bool
):

    prediction = model.predict([[
        age,
        bmi,
        children,
        is_female,
        is_smoker,
        region_southeast,
        bmi_category_Obese
    ]])

    return {
        "prediction": float(prediction[0])
    }