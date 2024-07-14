import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import streamlit as st

st.title("ML Training and Testing Split Interface")

# File uploader
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded DataFrame")
    st.write(df)

    # Sidebar for user inputs
    st.sidebar.header("User Input Parameters")

    def user_input_features():
        target_column = st.sidebar.selectbox("Select target column", df.columns)
        test_size = st.sidebar.slider('Test size', 0.1, 0.5, 0.2)
        random_state = st.sidebar.number_input('Random state', value=42)
        return target_column, test_size, random_state

    target_column, test_size, random_state = user_input_features()

    # Split the data into features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    # Apply one-hot encoding to categorical columns
    if len(categorical_cols) > 0:
        encoder = OneHotEncoder()
        X_encoded = encoder.fit_transform(X[categorical_cols])
        X_encoded_df = pd.DataFrame(X_encoded.toarray(), columns=encoder.get_feature_names_out(categorical_cols))
        X = X.drop(columns=categorical_cols)
        X = pd.concat([X, X_encoded_df], axis=1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train a sample model
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Display results
    st.subheader('Results')
    st.write('Accuracy:', accuracy_score(y_test, y_pred))

    # Display data splits
    st.subheader('Training and Testing Data Sizes')
    st.write('Training data size:', X_train.shape[0])
    st.write('Testing data size:', X_test.shape[0])

    st.subheader('Training Data Sample')
    st.write(X_train.head())

    st.subheader('Testing Data Sample')
    st.write(X_test.head())
else:
    st.write("Please upload a CSV file.")