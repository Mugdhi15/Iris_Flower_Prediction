# Iris_Flower_Prediction
This project is an interactive machine learning web application built with Streamlit to classify iris flowers into one of three species — Setosa, Versicolor, or Virginica — based on user-provided floral measurements.

The model is trained on the Iris dataset, one of the most famous datasets in machine learning, originally introduced by Sir Ronald Fisher. It contains 150 samples with 4 numeric features:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)
Each sample is labeled as one of the three iris species.

# Model Training
- A supervised classification algorithm was used for training — most likely a Random Forest Classifier (based on the use of model.feature_importances_). The training process involves:
- Splitting the dataset into training and testing sets
- Training the model on the training data
- Evaluating its performance using metrics like accuracy
- Saving the trained model as a serialized .pkl file using pickle

# Features of the App
- User Input Interface: Allows users to select values for all four input features using sliders.
- Real-time Prediction: Based on the inputs, the model predicts the most probable species of the flower.

Visual Feedback:
- Displays the predicted species and shows a corresponding flower image.
- Visualizes prediction probabilities across all species using a Plotly bar chart.
- Shows model feature importances to help users understand which features were most influential in making predictions.
- Transparency: The app displays the raw input data used for the prediction in an expandable section.
