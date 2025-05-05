import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
data = pd.read_csv('crop_recommendation.csv')

# Encode labels
tanaman = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
           'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
           'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple',
           'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']
for x in range(len(tanaman)):
    ganti = {tanaman[x]: x}
    data['label'] = data['label'].replace(ganti)

# Split data
Train_set, Val_set = train_test_split(data, test_size=0.2, random_state=40)
Test_set, Val_set = train_test_split(Val_set, test_size=0.5, random_state=40)

X_train, y_train = Train_set.drop(columns=['label']), Train_set['label']
X_val, y_val = Val_set.drop(columns=['label']), Val_set['label']
X_test, y_test = Test_set.drop(columns=['label']), Test_set['label']

# Train Random Forest model
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Save the model
joblib.dump(model_rf, "model_rf.pkl")
print("Model saved as model_rf.pkl")