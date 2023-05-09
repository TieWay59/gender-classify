import pickle
# import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# pickle DecisionTreeClassifier from version 1.0.2
model: DecisionTreeClassifier = pickle.load(open('../static/dt_model.pkl', 'rb'))

column_names = {
    'gender': ' Gender',
    'age': ' Age',
    'height_cm': ' Height (cm)',
    'weight_kg': ' Weight (kg)',
    'occupation': ' Occupation',
    'education_level': ' Education Level',
    'marital_status': ' Marital Status',
    'income_usd': ' Income (USD)',
    'favorite_color': ' Favorite Color'
}

print(model.predict([1, 2, 3]))
