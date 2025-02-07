import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def build_preprocessor():
    # Define preprocessing for numerical columns
    numeric_features = ["age", "bmi", "children"]
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Define preprocessing for categorical columns
    categorical_features = ["sex", "smoker", "region"]
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor
# In this part of our work we build some function so that we can preprocess our data. We will use this function in our model building part.
# This function will be used to preprocess the data before training the model.
# The function will be used to preprocess the data before making predictions with the model.
# By using this we avoid some of manual work and it will be easy to use this function in our model building part.