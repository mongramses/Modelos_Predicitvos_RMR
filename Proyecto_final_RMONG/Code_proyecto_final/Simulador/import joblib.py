import joblib

# Reemplaza con la ruta real al archivo
modelo = joblib.load("logistic_model.pkl")
print(type(modelo))


import joblib

joblib.dump(logreg, "logistic_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_names, "feature_names.pkl")
