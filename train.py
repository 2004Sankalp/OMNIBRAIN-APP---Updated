import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Loading Real-World Multi-Class Data...")

# 1. Load your uploaded dataset
df = pd.read_csv('heart_disease_uci.csv')

# 2. Separate the inputs (X) from the target disease severity (y)
# We drop 'id' and 'dataset' because they aren't medical vitals
X = df.drop(columns=['id', 'dataset', 'num'])
y = df['num']

# 3. Identify which columns are numbers and which are text/booleans
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object', 'bool']).columns

# Convert booleans to text so the AI can read them easily
X[cat_cols] = X[cat_cols].astype(str)

# 4. Create the "Magic" Cleaners (The Pipeline)
# This automatically fills missing numbers with the median, 
# and converts text like "Male" or "typical angina" into computer math.
numeric_transformer = SimpleImputer(strategy='median')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

# 5. Build and Train the Multi-Class Model!
print("Training the OMNIBRAIN AI Pipeline...")
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

clf.fit(X, y)

# 6. Save the entire pipeline (cleaning rules + AI model) to one file
joblib.dump(clf, 'heart_model.pkl')
print("Success! Model saved. You are ready for the presentation!")