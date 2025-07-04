import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.datasets import load_iris

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load iris feature names
iris = load_iris()
feature_names = iris.feature_names
target_names = iris.target_names

st.set_page_config(page_title="Iris Flower Prediction", layout="centered")

st.title("🌼 Iris Flower Prediction App")
st.write("Input flower measurements and get the predicted species with explanations.")

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.4)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.3)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Make prediction
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)

st.subheader("🌿 Prediction:")
st.write(f"The predicted species is: **{target_names[prediction]}**")

# Show predicted flower image
image_path_dict = {
    'setosa': 'images/setosa.png',
    'versicolor': 'images/versicolor.png',
    'virginica': 'images/virginica.png',
}

predicted_flower = target_names[prediction].lower()

if predicted_flower in image_path_dict:
    st.image(image_path_dict[predicted_flower], caption=f"Iris {predicted_flower.capitalize()}", use_container_width=True)


# For prediction probabilities
st.subheader("📊 Prediction Probabilities:")
prob_df = pd.DataFrame(prediction_proba, columns=target_names)
fig_prob = px.bar(prob_df.T, labels={'index': 'Class', 'value': 'Probability'})
fig_prob.update_layout(xaxis_tickangle=0)  # <-- sets x-axis labels horizontal
st.plotly_chart(fig_prob)

# For feature importance
st.subheader("🔍 Model Feature Importances:")
feat_imp = pd.Series(model.feature_importances_, index=feature_names)
fig_feat = px.bar(feat_imp.sort_values(ascending=False), labels={'index': 'Feature', 'value': 'Importance'})
fig_feat.update_layout(xaxis_tickangle=0)  # <-- sets x-axis labels horizontal
st.plotly_chart(fig_feat)

# Display raw input
with st.expander("🔎 Show Raw Input Data"):
    st.write(pd.DataFrame(input_data, columns=feature_names))
