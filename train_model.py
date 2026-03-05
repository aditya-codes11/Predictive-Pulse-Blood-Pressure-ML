import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_csv('patient_data.csv')
df.rename(columns={'C': 'Gender'}, inplace=True)

# Strip column names
df.columns = df.columns.str.strip()

# Convert ranges to mid
def range_to_mid(val):
    val = str(val).strip()
    if '+' in val:
        return int(val.replace('+', '').strip())
    elif '-' in val:
        nums = val.split('-')
        return (int(nums[0].strip()) + int(nums[1].strip())) // 2
    else:
        return pd.to_numeric(val, errors='coerce')

df['Systolic'] = df['Systolic'].apply(range_to_mid)
df['Diastolic'] = df['Diastolic'].apply(range_to_mid)

# Age mapping
age_mapping = {
    '18-25': 21.5,
    '26-35': 30.5,
    '36-45': 40.5,
    '46-55': 50.5,
    '56-65': 60.5,
    '65+': 70
}
df['Age'] = df['Age'].map(age_mapping)

# Drop duplicates
df.drop_duplicates(keep='first', inplace=True)

# Label encode
categorical_cols_for_label_encoding = ['Gender', 'Severity', 'History', 'Patient', 'TakeMedication', 'BreathShortness', 'VisualChanges', 'NoseBleeding', 'ControlledDiet', 'Stages']

label_encoder = LabelEncoder()
for col in categorical_cols_for_label_encoding:
    df[col] = label_encoder.fit_transform(df[col])

# Split
x = df.drop('Stages', axis=1)
y = df['Stages']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Train model
random_forest = RandomForestClassifier()
random_forest.fit(x_train, y_train)

# Save model
joblib.dump(random_forest, 'model.pkl')
print("Model saved successfully")