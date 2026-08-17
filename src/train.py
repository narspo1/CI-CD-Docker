import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset 
df = pd.read_csv("data/students.csv")

# Fetures
x = df[
    [
        "study_hours",
        "attendance",
        "previous_marks",
        "assignment_score"
    ]
]

# target
y = df["result"]

# Convert pass to fail 1/0
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split the dataset
x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# create Model 
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# train model
model.fit(x_train,y_train)

# Prediction
y_pred = model.predict(x_test)

# Moadel Accuracny
accurancy = accuracy_score(y_test,y_pred)
print("Model Accurancy:",accurancy)

# Save the model
import os
os.makedirs("model", exist_ok=True)
joblib.dump(
    {
        "model": model,
        "encoder": encoder
    },
    "model/model.pkl"
)
print("Model saved successfully")
