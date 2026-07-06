# Instagram Content Performance Prediction using Machine Learning
# Algorithm Used: Random Forest Classifier

# Step 1: Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
# Step 2: Load the Dataset

df = pd.read_csv("Instagram_Analytics.csv")

print("First 5 Rows of Dataset:")
print(df.head())

# Step 3: Explore the Dataset

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Step 4: Feature Selection
# Remove unnecessary columns

df = df.drop(
    ['post_id', 'account_id', 'post_datetime', 'post_date'],
    axis=1
)

print("\nColumns after Feature Selection:")
print(df.columns)

print("\nDataset Shape after Feature Selection:")
print(df.shape)

# Step 5: Check Target Class Distribution

print("\nPerformance Categories:")
print(df['performance_bucket_label'].value_counts())

# Step 6: Visualize Target Distribution

df['performance_bucket_label'].value_counts().plot(kind='bar')

plt.title("Performance Category Distribution")
plt.xlabel("Performance Category")
plt.ylabel("Number of Posts")

plt.show()

# Step 7: Label Encoding
# Convert categorical values into numbers

encoders = {}

categorical_columns = [
    'account_type',
    'media_type',
    'content_category',
    'traffic_source',
    'day_of_week',
    'performance_bucket_label'
]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

print("\nDataset after Label Encoding:")
print(df.head())

print("\nUpdated Dataset Information:")
print(df.info())

# Step 8: Separate Features (X) and Target (y)

X = df.drop('performance_bucket_label', axis=1)

y = df['performance_bucket_label']

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# Step 9: Split Dataset
# 80% Training
# 20% Testing

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Feature Shape:", X_train.shape)
print("Testing Feature Shape:", X_test.shape)

print("Training Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)

# Step 10: Train Random Forest Model

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")

# Step 11: Make Predictions

y_pred = model.predict(X_test)

# Step 12: Evaluate Model Accuracy

accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy*100:.2f}%")

# Step 13: Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.show()

# Step 14: Classification Report

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# Step 15: Feature Importance

feature_importance = pd.DataFrame({

    'Feature': X.columns,

    'Importance': model.feature_importances_

})

feature_importance = feature_importance.sort_values(

    by='Importance',

    ascending=False

)

print("\nFeature Importance:")

print(feature_importance)

# Step 16: Visualize Feature Importance

plt.figure(figsize=(12,6))

plt.bar(

    feature_importance['Feature'],

    feature_importance['Importance']

)

plt.xticks(rotation=90)

plt.xlabel("Features")

plt.ylabel("Importance")

plt.title("Feature Importance")

plt.tight_layout()

plt.show()

# Step 17: Save Trained Model

joblib.dump(model, "instagram_model.pkl")

print("\nMachine Learning Model Saved Successfully!")

# Step 18: Save Label Encoders

joblib.dump(encoders, "label_encoders.pkl")

print("Label Encoders Saved Successfully!")

print("\nProject Completed Successfully!")