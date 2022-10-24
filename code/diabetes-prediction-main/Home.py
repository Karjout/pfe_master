import streamlit as st
from PIL import Image
import time

st.set_page_config(
    page_title="Home",
    page_icon="👋",

)
with st.spinner('Wait for it...'):
    time.sleep(3)
image = Image.open('./img/uit.png')
st.image(image, caption='Ibn tofail university', use_column_width=True)
st.write("# Welcome to Diabte Application! 👋")

st.markdown(
    """
   In this modest work, we designed and developed a web application for the early prediction of **type 2**
   diabetes, in order to reduce the risk of complications of this disease on the patient’s health. To achieve this goal, we used algorithms supervised machine learning (K nearest neighbors, Decision Trees, Random Forest, Support Vector Machine, Natives Bayes) and the data set extracted from the hospital in Frankfurt (Germany). The performance of classifiers was compared based on accuracy rate and model sensitivity. The highest classification rates obtained by the application of Random Forest and the decision tree are respectively 91% and 87%, by applying the two methods of evaluation train /test and cross validation.
"""
)
st.write("# About Me 👋")
st.info(
    """
  Passionate about new technologies, I was able to acquire during my training the necessary theoretical and practical knowledge in Data Science, Business Intelligence, Machine Learning and Artificial Intelligence. Flexible in character, I adapt easily to new environments and I am also rigorous, dynamic and punctual.
"""
)
image = Image.open('./img/cv_pic.jpg')
st.image(image, caption='MR. Karjout Abdeslam')
st.markdown(
    """
    Linkden Profil : [Karjout Abdeslam](https://www.linkedin.com/in/karjout-abdeslam/)
    
    Github Profil : [Karjout Abdeslam](https://github.com/Karjout)
    
    My Personal Website : [Karjout.me](https://karjout.me//)
    
    Email : Karjout.abdeslam@gmail.com
    """
)
