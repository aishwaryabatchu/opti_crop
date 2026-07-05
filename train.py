import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Load Dataset
data = pd.read_csv('data/crop_data.csv')

# 2. Features and Target Split
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

# 3. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 4. Model Training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 5. Save the trained model
import os
os.makedirs('models', exist_ok=True)
with open('models/crop_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model Successfully Built and Saved inside models/ folder!")