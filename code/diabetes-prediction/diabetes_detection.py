from dl_model_pipeline import load_dl_model, load_scaler_dl
from ml_model_pipeline import load_ml_model, set_bmi, set_insulin, load_scaler_ml
from plot_performance import lst, plot_line

import streamlit as st
import pandas as pd
import numpy as np
import webbrowser

from PIL import Image
import plotly.graph_objs as go


ml_training = 'https://colab.research.google.com/drive/1z_-SRUKgo0fQ6d7AVu3uvJdeNvaw2ZsO?usp=sharing'
pycaret = 'https://colab.research.google.com/drive/1qwfNPrJOyB6NHYmt6gIYSNyQamXiphi5?usp=sharing'
kaggle = 'https://www.kaggle.com/risenattarach/deep-learning-prediction-val-acc-91-45'
learn_pycaret = 'https://www.kaggle.com/risenattarach/complete-beginners-guide-to-pycaret'


# get the data
data = pd.read_csv('diabetes.csv')


def main():

    my_page = st.sidebar.radio('Page Navigation', [
                               'Model Prediction', 'Technical Report', 'Modeling with Pycaret'])

    if my_page == 'Model Prediction':

        # create a title and sub-title
        st.write("""
        # Diabetes Detection using Machine Learning and Deep Learning
        Diabetes mellitus, commonly known as diabetes, is a **metabolic disease** that causes **high blood sugar**.
        The hormone insulin moves sugar from the blood into your cells to be stored or used for energy.
        With diabetes, your body either **doesn’t make enough insulin** or **can’t effectively use the insulin it does make**.

        """)
        image = Image.open('img/diabetes.jpg')
        st.image(image, caption='images from adobe stock',
                 use_column_width=True)

        st.write("""
        ### Objective
        Build machine learning and deep learning models to accurately predict whether or not the patients have diabetes.
        """)

        # set a subheader
        st.subheader('Data Information:')
        # show the data as a table
        st.dataframe(data)
        # show statistics on the data
        st.write(data.describe())
        # user input data for model prediction
        st.sidebar.title("Input your data for model prediction")
        user_data = []
        user_data.append(
            st.sidebar.number_input(
                label="Pregnancies",
                min_value=0,
                max_value=40,
                value=2,
                format="%i"
            ))
        user_data.append(
            st.sidebar.number_input(
                label="Glucose",
                min_value=0,
                max_value=400,
                value=119,
                format="%i"
            ))
        user_data.append(
            st.sidebar.number_input(
                label="BloodPressure",
                min_value=0,
                max_value=400,
                value=64,
                format="%i"))
        user_data.append(
            st.sidebar.number_input(
                label="SkinThickness",
                min_value=0,
                max_value=400,
                value=18,
                format="%i"))
        user_data.append(
            st.sidebar.number_input(
                label="Insulin",
                min_value=0,
                max_value=1600,
                value=92,
                format="%i"))
        user_data.append(
            st.sidebar.number_input(
                label="BMI",
                min_value=0.0,
                max_value=100.0,
                value=39.4,
                format="%f",
                step=1.0))
        user_data.append(
            st.sidebar.number_input(
                label="DiabetesPedigreeFunction",
                min_value=0.0,
                max_value=400.0,
                value=0.775,
                format="%f",
                step=1.0))
        user_data.append(
            st.sidebar.number_input(
                label="Age",
                min_value=0,
                max_value=150,
                value=23,
                format="%i"))

        st.write(f"""
        # Your Input data:

        **Please enter your data in the side bar and double check your data.**

        |Pregnancies| Glucose |  BloodPressure | SkinThickness | Insulin | BMI | Diabetes Pedigree Function | Age|
        |-----------|---------|----------------|---------------|---------|-----|----------------------------|----|
        |{user_data[0]}| {user_data[1]}| {user_data[2]}| {user_data[3]}| {user_data[4]}|  {user_data[5]} | {user_data[6]}|{user_data[7]}|
        \n """)

        st.write("\n")
        button = st.button("Predict")

        image = Image.open('img/diabetes2.jpg')
        st.image(image, caption='images from adobe stock',
                 use_column_width=True)

        # model prediction result for ML model
        def prediction_result_ml(pred, model_name):

            if np.argmax(pred, axis=1) == 1:
                st.write(f"**{model_name} Model Prediction:** \n")
                st.error("[Result] : You have risk of diabetes")
                st.error("[Confidence Level] : " +
                         str("{:.2f}".format(pred[0][1] * 100)) + "%")

            else:
                st.write(f"**{model_name} Model Prediction:** \n")
                st.success("[Result] : You are healthy")
                st.success("[Confidence Level] : " +
                           str("{:.2f}".format(pred[0][0] * 100)) + "%")

        # model prediction result for ANN model
        def prediction_result_dl(pred, model_name):

            if pred >= 0.5:
                st.write(f"**{model_name} Model Prediction:** \n")
                st.error('[Result] : You have high risk of diabetes')
                st.error("[Confidence Level] : " +
                         str("{:.2f}".format(pred[0][0] * 100)) + "%")
            else:
                st.write(f"**{model_name} Model Prediction:** \n")
                st.success("[Result] : You are healthy")
                st.success("[Confidence Level] : " +
                           str("{:.2f}".format((1 - pred[0][0]) * 100)) + "%")

        # preprocessing for ML model
        class Preprocessing:

            def __init__(self, user_data):
                self.user_data = user_data

            def feature_engineering(self, feat_en=False):
                if feat_en == True:
                    data = set_bmi(self.user_data)
                    data = set_insulin(data)
                    data = pd.get_dummies(data)
                return data

            def scaling_method(self, scaler, data_to_transform):
                num_col = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                           'BMI', 'DiabetesPedigreeFunction', 'Age']

                try:
                    scaled_data = scaler.transform(data_to_transform)
                except:
                    scaled_data = pd.DataFrame(scaler.transform(
                        data_to_transform[num_col]), columns=data_to_transform[num_col].columns, index=data_to_transform[num_col].index)
                    bin_col = data_to_transform.drop(columns=num_col, axis=1)
                    scaled_data = bin_col.merge(
                        scaled_data, left_index=True, right_index=True, how="right")

                return scaled_data

        class ModelPrediction:

            def __init__(self, model, name):
                self.model = model
                self.name = name

            def predict(self, user_data, deepLearning=False):

                if deepLearning == True:
                    result = self.model.predict(user_data, verbose=0)
                    prediction_result_dl(result, self.name)
                else:
                    result = self.model.predict_proba(user_data)
                    prediction_result_ml(result, self.name)

        if button:
            # get record from user
            user_record = {}
            features = ['Pregnancies', 'Glucose', 'BloodPressure',
                        'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            # append user data to dict
            for f, b in zip(features, user_data):
                user_record[f] = b
            # create dataframe based on user input data for model prediction
            user_data = pd.DataFrame(user_record, index=[0])

            # load ANN model
            nn_model = load_dl_model()
            # load scaler (Model of ML and DL were trained differently)
            min_max = load_scaler_dl()
            # data preprocessing
            data_dl = Preprocessing(user_data)
            data_dl_scaled = data_dl.scaling_method(min_max, user_data)
            # pass in loaded model and named it
            dl_pred = ModelPrediction(nn_model, "Deep Learning")
            # model prediction result for DL model
            dl_pred.predict(data_dl_scaled, deepLearning=True)

            # load ML models
            lgr, svc, dt, rdf, ada, lgbm = load_ml_model()
            # load scaler (Model of ML and DL were trained differently)
            min_max_scaler, standard_scaler = load_scaler_ml()
            # pass in user_data
            df = Preprocessing(user_data)
            # apply standard scaler to user_data (w/o feature engineering)
            df_sd = df.scaling_method(standard_scaler, user_data)
            # apply min_max scaler to user_data (w/o feature engineering)
            df_mm = df.scaling_method(min_max_scaler, user_data)

            logisticReg = ModelPrediction(lgr, "LogisticRegression")
            logisticReg.predict(df_mm)

#             KNN = ModelPrediction(knn, "KNN")
#             KNN.predict(df_sd)

            supportVecM = ModelPrediction(svc, "SVC")
            supportVecM.predict(df_sd)

            DecisionTree = ModelPrediction(dt, "DecisionTree")
            DecisionTree.predict(user_data)

            Adaboost = ModelPrediction(ada, "AdaBoost")
            Adaboost.predict(user_data)

            RandomForest = ModelPrediction(rdf, "RandomForest")
            RandomForest.predict(user_data)

            LGBM = ModelPrediction(lgbm, "LGBM")
            LGBM.predict(user_data)

    elif my_page == 'Technical Report':

        st.write("""
        # Technical Report
        ## Data
        The datasets consist of several medical predictor (independent variables) and one target (dependent) variable. Independent variables include **Pregnancies**, **Glucose**, **BloodPressure**, **SkinThickness**, **Insulin**, **BMI**, **DiabetesPedigreeFunction**, **Age**. Dependent variable includes **Outcome**.
        [link to the data](https://www.kaggle.com/uciml/pima-indians-diabetes-database)""")

        image = Image.open('img/diabetes_web.jpg')
        st.image(image, caption='image from adobe stock',
                 use_column_width=True)

        st.write("""
                ## Columns
                |Columns|Description|
                |-------|------------|
                |Pregnancies|Number of times pregnant|
                |Glucose|Plasma glucose concentration for 2 hours in an oral glucose tolerance test|
                |BloodPressure|Diastolic blood pressure (mm Hg)|
                |SkinThickness|Triceps skin fold thickness (mm)|
                |Insulin|2-Hour serum insulin (mu U/ml)|
                |BMI|Body mass index (weight in kg/(height in m)^2)|
                |DiabetesPedigreeFunction|Diabetes pedigree function|
                |Age|Age (years)|
                |Outcome|Class variable (0 or 1) 268 of 768 are 1, the others are 0|""")

        st.write("""
        ## Model Performance

        **Confusion matrix** : also known as the error matrix, allows visualization of the performance of an algorithm :

        - True Positive (TP): Diabetic, correctly identified as diabetic
        - True Negative (TN): Healthy, correctly identified as healthy
        - False Positive (FP): Healthy, incorrectly identified as diabetic
        - False Negative (FN): Diabetic, incorrectly identified as healthy
        ### Metrics

        - Accuracy : (TP +TN) / (TP + TN + FP +FN)
        - Precision : TP / (TP + FP)
        - Recall : TP / (TP + FN)
        - F1 score : 2 x ((Precision x Recall) / (Precision + Recall))
        - Roc Curve : The ROC curve is created by plotting the true positive rate (TPR) against the false positive rate (FPR) in various threshold settings.

        **Precision Recall Curve** : shows the tradeoff between precision and recall for different thresholds
        To train and test our algorithm we'll use cross validation K-Fold

        In **K-fold cross-validation**, the original sample is randomly partitioned into K equal sized subsamples. Of the k subsamples, a single subsample is retained as the validation data for testing the model, and the remaining K − 1 subsamples are used as training data. The cross-validation process is then repeated K times, with each of the k subsamples used exactly once as validation data. The k results can then be averaged to produce a single estimation. The advantage of this method over repeated random sub-sampling is that all observations are used for both training and validation, and each observation is used for validation exactly once.
            
        """)

        st.write("""
                ## Correlation pair plot""")

        # open img
        image = Image.open('img/sns.pairplot.png')
        st.image(image, caption='correlation-plot',
                 use_column_width=True)

        # paging
        selectbox = st.selectbox(
            'Select Model', ['Deep Learning Model', 'Machine Learning Model'])

        if selectbox == 'Deep Learning Model':
            # button to url
            """[Deep Learning Modeling Part](https://www.kaggle.com/risenattarach/deep-learning-prediction-val-acc-91-45)"""

            # open image
            st.write("""
                ## Model Architecture""")
            image = Image.open('img/model_architect.PNG')
            st.image(image, caption='model-architecture',
                     use_column_width=True)

            # create subheader
            st.subheader('Model Training Info (Training set)')
            # dataframe of model training info
            model_logs_ex = pd.read_csv('dl_model_training/my_logs4.csv')

            # color of plots
            color = ['red', 'green', 'black', 'blue', 'pink']
            # plot line corresponding to the color and column in dataframe
            for i, c in zip(model_logs_ex.columns[1:6], color):
                plot_line(model_logs_ex, i, str(i), c)
            # combine all line plot for training set
            fig1 = go.Figure(data=lst[0]+lst[1]+lst[2]+lst[3]+lst[4])
            st.plotly_chart(fig1)

            st.subheader('Model Training Info (Validation set)')
            # color of plots
            color = ['red', 'green', 'black', 'blue', 'pink']
            # plot line corresponding to the color and column in dataframe
            for i, c in zip(model_logs_ex.columns[6:11], color):
                plot_line(model_logs_ex, i, str(i), c)
            # combine all line plot for evaluation set
            fig2 = go.Figure(data=lst[5]+lst[6]+lst[7]+lst[8]+lst[9])
            st.plotly_chart(fig2)

            # logs info
            st.write("""
                ## Logs Information""")
            st.write(model_logs_ex)

        elif selectbox == 'Machine Learning Model':
            """[Machine Learning Modeling Part](https://colab.research.google.com/drive/1z_-SRUKgo0fQ6d7AVu3uvJdeNvaw2ZsO?usp=sharing)"""
            image = Image.open('img/diabetes_end.png')
            st.image(image, caption='images from adobe stock',
                     use_column_width=True)

    elif my_page == "Modeling with Pycaret":
        st.write("""
                ## Pycaret Modeling""")
        # set a subheader and display the users input
        st.subheader(
            'Model Perfomance: Before applying Feature Engineering (Note: target leakage not fixed)')
        # display an image
        image = Image.open('pycaret-screenshots/base-model-acc.JPG')
        st.image(image, caption='model', use_column_width=True)
        # set a subheader and display the users input
        st.subheader(
            'Model Perfomance: After applied Feature Engineering (Note: target leakage not fixed)')
        # display an image
        image = Image.open(
            'pycaret-screenshots/after-feature-engineering-Pycaret.JPG')
        st.image(image, caption='model', use_column_width=True)

        """[Pycaret Modeling](https://colab.research.google.com/drive/1qwfNPrJOyB6NHYmt6gIYSNyQamXiphi5?usp=sharing)"""
        """[Learn more about Pycaret](https://www.kaggle.com/risenattarach/complete-beginners-guide-to-pycaret)"""


if __name__ == "__main__":
    main()
