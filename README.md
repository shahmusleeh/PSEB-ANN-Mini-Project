# Bank Customer Churn Predictor (ANN + Streamlit)

An end-to-end Machine Learning and Deep Learning web application that predicts whether a bank customer is likely to stay or leave (churn) using an Artificial Neural Network (ANN) built with TensorFlow/Keras and deployed via Streamlit.

---

## Overview
Customer churn—when clients stop doing business with a company—is a critical metric in banking. Using the `Churn_Modelling.csv` dataset (10,000 customer records), this project implements a full supervised learning pipeline:
1. **Exploratory Data Analysis & Cleaning:** Removing non-predictive identifier columns (`RowNumber`, `CustomerId`, `Surname`) and analyzing the ~80/20 class distribution.
2. **Stratified Train-Test Split:** Splitting data into 8,000 training and 2,000 test records while preserving the ~20.4% churn rate.
3. **Feature Preprocessing:** Applying `OneHotEncoder` to categorical variables (`Geography`, `Gender`) and `StandardScaler` to numerical features via `ColumnTransformer` (fitted exclusively on training data to prevent data leakage).
4. **ANN Architecture:** A sequential neural network with an input layer (13 features), two ReLU hidden layers (16 and 8 neurons), and a Sigmoid output layer for binary probability estimation.
5. **Interactive Web App:** A user-friendly Streamlit dashboard (`app.py`) for real-time customer churn predictions.

---

## Repository Structure

```text
├── Churn_Modelling.csv            # Dataset (10,000 customer records)
├── churn_predictor_pipeline.py    # Complete training, evaluation, and artifact export script
├── app.py                         # Streamlit web application
├── model.keras                    # Saved trained Keras ANN model
├── preprocessor.pkl               # Saved scikit-learn ColumnTransformer
├── loss_plot.png                  # Training vs. validation loss curve
├── requirements.txt               # Python dependencies for local & Streamlit Cloud deployment
└── README.md                      # Project documentation
```

---

##  Getting Started

### 1. Local Setup & Installation
Clone this repository and install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Train the Model (Optional)
If you want to retrain the ANN from scratch and regenerate `model.keras`, `preprocessor.pkl`, and `app.py`:
```bash
python churn_predictor_pipeline.py
```

### 3. Run the Streamlit App Locally
Launch the web interface in your browser:
```bash
streamlit run app.py
```

---

## Model Architecture & Performance

### Neural Network Structure
* **Input Layer:** 13 neurons (8 scaled numerical features + 3 One-Hot encoded `Geography` columns + 2 One-Hot encoded `Gender` columns)
* **Hidden Layer 1:** 16 neurons, `ReLU` activation
* **Hidden Layer 2:** 8 neurons, `ReLU` activation
* **Output Layer:** 1 neuron, `Sigmoid` activation
* **Optimizer & Loss:** `Adam` optimizer with `binary_crossentropy` loss and `EarlyStopping` (patience = 5)

### Evaluation Metrics (Test Set: 2,000 Customers)
* **Test Accuracy:** `86.15%` (outperforming the ~79.6% naive baseline)
* **Class-1 Recall (Churners):** `0.49`
* **Confusion Matrix:**
  * True Negatives (Stayed predicted correctly): `1,522`
  * False Positives (Stayed predicted as Left): `71`
  * False Negatives (Left predicted as Stayed): `206`
  * True Positives (Left predicted correctly): `201`

---

## Sample Test Cases
You can test the deployed app using these two profiles:
* **Likely to STAY:** Age `38`, Geography `France`, Gender `Male`, Active Member `Yes`, Balance `60,000`, Credit Score `650`, Number of Products `2`.
* **Likely to LEAVE:** Age `60`, Geography `Germany`, Gender `Female`, Active Member `No`, Balance `120,000`, Credit Score `600`, Number of Products `1`.

---

## Deploying on Streamlit Community Cloud
1. Push `app.py`, `model.keras`, `preprocessor.pkl`, and `requirements.txt` to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io/), sign in with GitHub, and click **New app**.
3. Select your repository, branch, and set the main file path to `app.py`.
4. Click **Deploy!**
