import pandas as pd

def preprocess_input(input_dict, feature_columns):
    """
    Preprocesses raw user input into a format suitable for the trained model.
    """
    # Create a dataframe
    df_raw = pd.DataFrame([input_dict])
    
    # Drop unused columns
    df_raw = df_raw.drop(columns=['gender'], errors='ignore')
    
    # Bin tenure
    bins = [-1, 12, 24, 48, 72]
    labels = ['0-12', '13-24', '25-48', '49-72']
    df_raw['tenure_group'] = pd.cut(df_raw['tenure'], bins=bins, labels=labels)
    
    # Get dummies
    categorical_cols = df_raw.select_dtypes(include=['object', 'category']).columns.tolist()
    df_encoded = pd.get_dummies(df_raw, columns=categorical_cols, drop_first=True)
    
    # Reindex to match the exact model training columns
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)
    
    # Cast booleans to float
    for col in df_encoded.select_dtypes(include=['bool']).columns:
        df_encoded[col] = df_encoded[col].astype(float)
        
    return df_encoded
