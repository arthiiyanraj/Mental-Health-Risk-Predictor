import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv(r"C:\Users\aadhav\OneDrive\Desktop\ml project\mental_health(1).csv")  # Use the uploaded CSV

le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

X = df.drop("Mental_Health_Risk", axis=1)
y = df["Mental_Health_Risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("training set size:" ,X_train.shape[0])
print("testing set size:" ,X_test.shape[0])

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

joblib.dump(model, 'mental_health_model.pkl')
