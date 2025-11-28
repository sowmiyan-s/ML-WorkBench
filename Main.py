import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import numpy as np
import io, zipfile, pickle

x_cols = []
y_col = None

st.title("ML Workbench")

#--------------------------------------------
# Sidebar - About App
st.sidebar.title("About App")
st.sidebar.info(
    "**ML Workbench** is an enterprise-grade data analysis and machine learning platform designed to democratize AI. "
    "It empowers users to seamlessly upload datasets, perform robust preprocessing, train state-of-the-art models, "
    "and derive actionable insights through an intuitive, code-free interface."
)
st.sidebar.markdown("---")
st.sidebar.link_button("View on GitHub", "https://github.com/sowmiyan-s/ML-WorkBench")
st.sidebar.markdown("Created by [Sowmiyan S](https://github.com/sowmiyan-s)")

#--------------------------------------------
# Upload Dataset
# Upload Dataset
st.header("Step 1: Upload Your Data")
st.markdown("Start by uploading your CSV file. This is the data we will use to train the model.")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of dataset")
    st.write(df.head())
    st.write("Shape:", df.shape)

#--------------------------------------------
# Preprocessing
# Preprocessing
if uploaded_file:
    st.header("Step 2: Clean and Prepare Data")
    st.markdown("Data often needs cleaning before it can be used. Use the options below to fix common issues.")

    # Drop missing values
    if st.checkbox("Drop rows with missing values"):
        df.dropna(inplace=True)
        st.write("After dropping NA:", df.shape)

    # Identify numeric and categorical columns
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    st.markdown("### Data Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Numeric Columns** ({len(num_cols)})")
        st.caption("Columns containing numbers.")
        st.write(num_cols)
    
    with col2:
        st.markdown(f"**Categorical Columns** ({len(cat_cols)})")
        st.caption("Columns containing text or categories.")
        st.write(cat_cols)

    with st.expander("View Detailed Statistics"):
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Missing Values': df.isnull().sum(),
            'Unique Values': df.nunique()
        })
        st.dataframe(col_info, use_container_width=True)

    # Normalize numeric columns
    normalize = st.multiselect("Scale Numbers (Optional)", num_cols, help="Adjusts numeric values to a common scale. Useful for some algorithms.")
    if normalize:
        from sklearn.preprocessing import StandardScaler
        df[normalize] = StandardScaler().fit_transform(df[normalize])

#--------------------------------------------
# Select Features and Target
if uploaded_file:
    all_cols = df.columns.tolist()
    st.header("Step 3: Choose What to Predict")
    st.markdown("Select the columns you want the model to learn from (Inputs) and the column you want to predict (Target).")
    x_cols = st.multiselect("Select Input Columns (Features)", all_cols, help="Choose the columns the model should learn from.")
    y_col = st.selectbox("Select Target Column (Prediction)", all_cols, help="Choose the column you want to predict.")

#--------------------------------------------
# Select Algorithm and Test Size
if x_cols and y_col:
    st.header("Step 4: Choose a Learning Method")
    st.markdown("Select an algorithm to train your model. If you're unsure, try **Random Forest**.")
    algo = st.selectbox("Select Method", [
        "Linear Regression", "Random Forest Regressor", "KNN Regressor", "SVR",
        "Logistic Regression", "Decision Tree", "Random Forest Classifier", "KNN Classifier", "SVM", "Naive Bayes"
    ])
    st.caption("Note: Regressors are for predicting numbers (e.g., price), Classifiers are for predicting categories (e.g., yes/no).")
    test_size = st.slider("Test Data Size (Fraction)", 0.1, 0.5, 0.2, help="How much data should be kept aside for testing? 0.2 means 20%.")

#--------------------------------------------
# Train Model
if st.button("Start Training", type="primary"):
    X = df[x_cols].values
    y = df[y_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Select model
    task = None
    if algo == "Linear Regression":
        model = LinearRegression(); task = "regression"
    elif algo == "Random Forest Regressor":
        model = RandomForestRegressor(); task = "regression"
    elif algo == "KNN Regressor":
        model = KNeighborsRegressor(); task = "regression"
    elif algo == "SVR":
        model = SVR(); task = "regression"
    elif algo == "Logistic Regression":
        model = LogisticRegression(max_iter=1000); task = "classification"
    elif algo == "Decision Tree":
        model = DecisionTreeClassifier(); task = "classification"
    elif algo == "Random Forest Classifier":
        model = RandomForestClassifier(); task = "classification"
    elif algo == "KNN Classifier":
        model = KNeighborsClassifier(); task = "classification"
    elif algo == "SVM":
        model = SVC(); task = "classification"
    elif algo == "Naive Bayes":
        model = GaussianNB(); task = "classification"

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.success("Model trained!")
    if task == "regression":
        st.write("MSE:", mean_squared_error(y_test, y_pred))
    else:
        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.text(classification_report(y_test, y_pred))

    # Save model in session
    st.session_state.model = model
    st.session_state.x_cols = x_cols
    st.session_state.y_col = y_col
    st.session_state.task = task

#--------------------------------------------
# Test Model with Custom Input
if "model" in st.session_state:
    st.subheader("Test Model with Custom Input")

    # Create input fields for all features
    user_input = []
    st.write("Enter values for features:")
    for col in st.session_state.x_cols:
        val = st.text_input(f"{col}", "")
        if val != "":
            try:
                val = float(val)
            except:
                st.warning(f"Invalid input for {col}, using 0")
                val = 0
        else:
            val = 0
        user_input.append(val)

    user_input_array = np.array(user_input).reshape(1, -1)

    # Predict button
    if st.button("Predict"):
        try:
            prediction = st.session_state.model.predict(user_input_array)
            if st.session_state.task == "regression":
                st.success(f"Predicted value: {prediction[0]:.4f}")
            else:
                st.success(f"Predicted class: {prediction[0]}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # Optional: prediction history
    if "pred_history" not in st.session_state:
        st.session_state.pred_history = []

    if st.button("Add to History"):
        try:
            prediction = st.session_state.model.predict(user_input_array)
            st.session_state.pred_history.append(prediction[0])
            st.write("Prediction history:", st.session_state.pred_history)
        except:
            st.error("Cannot add to history, prediction failed.")

#--------------------------------------------
# Export Model
if "model" in st.session_state:
    st.subheader("Export Model")
    save_path = st.text_input("Enter path to save ZIP", os.path.join(os.getcwd(), "trained_model.zip"))

    if st.button("Save Model ZIP"):
        try:
            with zipfile.ZipFile(save_path, "w") as zf:
                # Save model
                model_bytes = io.BytesIO()
                pickle.dump(st.session_state.model, model_bytes)
                zf.writestr("model.pkl", model_bytes.getvalue())
                # Save metadata
                info = f"X columns: {st.session_state.x_cols}\nY column: {st.session_state.y_col}\nTask: {st.session_state.task}"
                zf.writestr("model_info.txt", info)
            st.success(f"Model saved as {save_path}")
        except Exception as e:
            st.error(f"Failed to save: {e}")
