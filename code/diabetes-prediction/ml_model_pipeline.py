import joblib

# load trained model


def load_ml_model():

    lgr = joblib.load('model_updates/lgr_mm.sav')
#     knn = joblib.load('model_updates/knn_sd.sav')
    svc = joblib.load('model_updates/svc_sd.sav')
    dt = joblib.load('model_updates/dt.sav')
    rdf = joblib.load('model_updates/rdf.sav')
    ada = joblib.load('model_updates/ada.sav')
    lgbm = joblib.load('model_updates/lgbm.sav')

    return lgr, svc, dt, rdf, ada, lgbm

# feature engineering process 1


def set_bmi(row):

    row.loc[:, 'BM_DESC_Healthy'] = 0
    row.loc[:, 'BM_DESC_Obese'] = 0
    row.loc[:, 'BM_DESC_Over'] = 0
    row.loc[:, 'BM_DESC_Under'] = 0

    if row['BMI'].values < 18.5:
        row.loc[row['BMI'] < 18.5, 'BM_DESC_Under'] = 1
    elif (row["BMI"].values >= 18.5) and (row["BMI"].values <= 24.9):
        row.loc[(row["BMI"] >= 18.5) & (
            row["BMI"] <= 24.9), 'BM_DESC_Healthy'] = 1
    elif (row["BMI"] >= 25).values and (row["BMI"].values <= 29.9):
        row.loc[(row["BMI"] >= 25) & (row["BMI"] <= 29.9), 'BM_DESC_Over'] = 1
    elif row["BMI"].values >= 30:
        row.loc[row["BMI"] >= 30, 'BM_DESC_Obese'] = 1

    return row

# feature engineering process 2


def set_insulin(row):
    row.loc[:, 'INSULIN_DESC_Abnormal'] = 0
    row.loc[:, 'INSULIN_DESC_Normal'] = 0
    if (row["Insulin"].values >= 16) and (row["Insulin"].values <= 166):
        row.loc[(row["Insulin"] >= 16) & (
            row["Insulin"] <= 166), 'INSULIN_DESC_Normal'] = 1
    else:
        row.loc[:, 'INSULIN_DESC_Abnormal'] = 1

    return row

# load scaler for machine learning model


def load_scaler_ml():

    min_max_scaler = joblib.load(
        'model_updates/MinMaxScaler().pkl')
    standard_scaler = joblib.load(
        'model_updates/StandardScaler().pkl')

    return min_max_scaler, standard_scaler
