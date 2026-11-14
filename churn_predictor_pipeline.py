"""
ANN Mini-Project: Customer Churn Predictor Pipeline
Complete Python implementation covering Steps 0 through 9 from the project manual,
including "My understanding" notes in comments.
"""

# ==============================================================================
# Step 0: Setup
# ==============================================================================
# Required libraries: tensorflow, pandas, scikit-learn, joblib, streamlit, matplotlib
# Install via: pip install tensorflow pandas scikit-learn joblib streamlit matplotlib

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("TensorFlow Version:", tf.__version__)

# My understanding (Step 0):
# Setting up the environment and importing these libraries gives us the tools needed for
# data handling (pandas/numpy), preprocessing and evaluation (scikit-learn), deep learning
# (TensorFlow/Keras), plotting (matplotlib), and model saving/deployment (joblib/streamlit).


# ==============================================================================
# Step 1: Load the data
# ==============================================================================
# Set random seeds for reproducibility
tf.keras.utils.set_random_seed(42)
np.random.seed(42)

# Read the dataset
df = pd.read_csv('Churn_Modelling.csv')

print("\n--- Step 1: Dataset Shape & Preview ---")
print("Shape:", df.shape)
print(df.head())

# My understanding (Step 1):
# A DataFrame is a 2D table where each of the 10,000 rows represents a unique bank customer
# and each of the 14 columns represents a specific attribute or outcome for that customer.


# ==============================================================================
# Step 2: Explore and clean
# ==============================================================================
print("\n--- Step 2: Data Info & Target Distribution ---")
print(df.info())
print("\nExited Value Counts:")
print(df['Exited'].value_counts())
print("\nExited Proportions:")
print(df['Exited'].value_counts(normalize=True))

# Drop non-predictive identifier/name columns
df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
print("\nColumns after dropping IDs:", df.columns.tolist())

# My understanding (Step 2):
# We drop RowNumber, CustomerId, and Surname because arbitrary IDs and names hold no
# generalizable predictive value and would cause overfitting. The 80/20 split means a lazy
# model guessing "stay" gets ~80% accuracy, so our ANN must beat this baseline.


# ==============================================================================
# Step 3: Split into X, y, train and test
# ==============================================================================
X = df.drop(columns=['Exited'])
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("\n--- Step 3: Train/Test Split Shapes ---")
print("X_train shape:", X_train.shape, "| y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape, "| y_test shape:", y_test.shape)

# My understanding (Step 3):
# Keeping an unseen test set lets us evaluate how well the model generalizes to new customers
# rather than memorizing training examples. Stratifying ensures both train and test sets
# maintain the exact same ~20% churn ratio.


# ==============================================================================
# Step 4: Preprocess (encode + scale)
# ==============================================================================
numeric_cols = [
    'CreditScore', 'Age', 'Tenure', 'Balance',
    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
]
categorical_cols = ['Geography', 'Gender']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Golden rule: fit_transform on TRAIN, transform only on TEST
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\n--- Step 4: Processed Feature Shapes ---")
print("X_train_processed shape:", X_train_processed.shape)
print("X_test_processed shape:", X_test_processed.shape)

# My understanding (Step 4):
# Neural networks require scaled numbers so large values like Balance don't dominate smaller
# features like Tenure during gradient updates. We never fit the preprocessor on test data to
# avoid data leakage, ensuring test evaluation mimics real-world unseen data.


# ==============================================================================
# Step 5: Build the ANN
# ==============================================================================
model = keras.Sequential([
    keras.Input(shape=(X_train_processed.shape[1],)),
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n--- Step 5: Model Summary ---")
model.summary()

# My understanding (Step 5):
# ReLU introduces non-linearity so hidden layers can learn complex feature interactions, while
# Sigmoid squashes the final output into a probability between 0 and 1. The output layer has
# 1 neuron because we are solving a binary classification problem (churn vs. stay).


# ==============================================================================
# Step 6: Train the model
# ==============================================================================
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

print("\n--- Step 6: Training Model ---")
history = model.fit(
    X_train_processed, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Plot training and validation loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Binary Cross-Entropy Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_plot.png')
plt.close()
print("Saved training/validation loss plot to 'loss_plot.png'.")

# My understanding (Step 6):
# A batch is a subset of 32 samples used per weight update, and an epoch is one full pass
# through the training data. Comparing training and validation loss on the plot shows whether
# the model is learning steadily or starting to overfit (when validation loss rises).


# ==============================================================================
# Step 7: Evaluate on the test set
# ==============================================================================
print("\n--- Step 7: Test Set Evaluation ---")
y_pred_probs = model.predict(X_test_processed)
y_pred = (y_pred_probs >= 0.5).astype(int).flatten()

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Test Accuracy: {acc:.4f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

# My understanding (Step 7):
# The confusion matrix breaks down true/false positives and negatives, showing that while overall
# accuracy is ~86%, class-1 recall (~0.49) means we miss half of departing customers. A bank
# cares deeply about recall because failing to identify churning customers loses revenue.


# ==============================================================================
# Step 8: Save the model and the preprocessor
# ==============================================================================
model.save('model.keras')
joblib.dump(preprocessor, 'preprocessor.pkl')
print("\n--- Step 8: Saved Artifacts ---")
print("Saved 'model.keras' and 'preprocessor.pkl'.")

# Sanity check with one customer from test set
sample_customer = X_test.iloc[[0]]
actual_label = y_test.iloc[0]
sample_processed = preprocessor.transform(sample_customer)
sample_prob = model.predict(sample_processed)[0][0]
sample_pred = int(sample_prob >= 0.5)

print(f"Sanity Check -> Predicted Prob: {sample_prob:.4f} | Predicted Class: {sample_pred} | Actual Class: {actual_label}")

# My understanding (Step 8):
# We must save the fitted preprocessor alongside the model so new app inputs receive the exact
# same encoding and scaling transformations used during training. Without it, raw inputs would
# have the wrong format and scale, causing prediction errors or garbage outputs.


# ==============================================================================
# Step 9: Streamlit Web App Code (Save separately as app.py and run: streamlit run app.py)
# ==============================================================================
APP_PY_CODE = """
import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf

st.set_page_config(page_title="Customer Churn Predictor", page_icon="🏦", layout="centered")

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model('model.keras')
    preprocessor = joblib.load('preprocessor.pkl')
    return model, preprocessor

model, preprocessor = load_artifacts()

st.title("🏦 Bank Customer Churn Predictor")
st.markdown("Predict whether a bank customer will stay or leave using an Artificial Neural Network.")

st.sidebar.header("Customer Details")

credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 92, 38)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)
balance = st.sidebar.number_input("Account Balance", 0.0, 250000.0, 60000.0)
num_of_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.sidebar.selectbox("Has Credit Card", ["Yes", "No"])
is_active_member = st.sidebar.selectbox("Is Active Member", ["Yes", "No"])
estimated_salary = st.sidebar.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

input_df = pd.DataFrame([{
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': 1 if has_cr_card == "Yes" else 0,
    'IsActiveMember': 1 if is_active_member == "Yes" else 0,
    'EstimatedSalary': estimated_salary
}])

st.subheader("Input Summary")
st.write(input_df)

if st.button("Predict"):
    processed = preprocessor.transform(input_df)
    prob = model.predict(processed)[0][0]
    st.metric("Churn Probability", f"{prob * 100:.2f}%")
    if prob >= 0.5:
        st.error("Verdict: Customer is **likely to LEAVE**.")
    else:
        st.success("Verdict: Customer is **likely to STAY**.")
"""

with open("app.py", "w") as f:
    f.write(APP_PY_CODE.strip() + "\n")

print("Generated 'app.py' for Step 9.")

# My understanding (Step 9):
# When the user clicks Predict, the app collects all widget values into a single-row DataFrame,
# transforms it using the loaded preprocessor, feeds the processed vector into the ANN to get a
# probability, and displays the percentage along with a LEAVE (>=0.5) or STAY (<0.5) verdict.
