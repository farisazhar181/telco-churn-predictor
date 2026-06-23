def predict_churn(model, df_encoded):
    """
    Predicts the churn probability and risk level for a given processed dataframe.
    """
    prob = model.predict_proba(df_encoded)[0][1]
    
    risk_label = "Low"
    color = "green"
    if prob >= 0.6:
        risk_label = "High"
        color = "red"
    elif prob >= 0.3:
        risk_label = "Medium"
        color = "orange"
        
    return prob, risk_label, color
