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

x_cols = []
y_col = None


st.title("Ml Workbench")


st.subheader("Upload Dataset")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of dataset")
    st.write(df.head())
    st.write("Shape:", df.shape)



#----------------------------------------------------------------------------------------------

if uploaded_file:
    st.subheader("Preprocessing")

    # Drop missing values
    if st.checkbox("Drop rows with missing values"):
        df.dropna(inplace=True)
        st.write("After dropping NA:", df.shape)

    # Identify numeric and categorical columns
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    st.write("Numeric columns:", num_cols)
    st.write("Categorical columns:", cat_cols)

    # Normalize numeric columns
    normalize = st.multiselect("Normalize numeric columns", num_cols)
    if normalize:
        from sklearn.preprocessing import StandardScaler
        df[normalize] = StandardScaler().fit_transform(df[normalize])


#--------------------------------------------------------------------------------------------------


if uploaded_file:
    all_cols = df.columns.tolist()
    x_cols = st.multiselect("Select features (X)", all_cols)
    y_col = st.selectbox("Select target (Y)", all_cols)

#--------------------------------------------------------------------------------------------------



if x_cols and y_col:
    algo = st.selectbox("Choose Algorithm", [
        "Linear Regression", "Random Forest Regressor", "KNN Regressor", "SVR",
        "Logistic Regression", "Decision Tree", "Random Forest Classifier", "KNN Classifier", "SVM", "Naive Bayes"
    ])
    test_size = st.slider("Test size fraction", 0.1, 0.5, 0.2)


#--------------------------------------------------------------------------------------------------

if st.button("Train Model"):
    

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
    
    
    
    
#------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------
# Test model with custom input
if "model" in st.session_state:
    st.subheader("Test Model with Custom Input")

    import numpy as np

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

    # Button to predict
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

    
    
    
#--------------------------------------------------------------------------------------------------


if "model" in st.session_state:
    st.subheader("Export Model")
    save_path = st.text_input("Enter path to save ZIP", os.path.join(os.getcwd(), "trained_model.zip"))

    if st.button("Save Model ZIP"):
        import io, zipfile, pickle
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

