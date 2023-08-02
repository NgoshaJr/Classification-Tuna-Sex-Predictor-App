import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

# Function to remove outliers using the IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df.loc[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_filtered

# Load the data and remove the "Unnamed: 0" column
df = pd.read_csv('tunadata_cleaned.csv')
df = df.drop(columns=['Unnamed: 0'])

# Sidebar
st.sidebar.header("Dataset Summary")
st.sidebar.subheader("Dataset dimensions")
st.sidebar.write(f"Rows: {df.shape[0]}")
st.sidebar.write(f"Columns: {df.shape[1]}")

# Show the first five rows of the dataset
st.sidebar.subheader("First five rows of the dataset")
st.sidebar.write(df.head())

# Get the features and target
X = df[['gonadalWeight', 'focalLength']]
y = df['Sex']

# Remove outliers from 'gonadalWeight' and 'focalLength' columns
X = remove_outliers(X, 'gonadalWeight')
X = remove_outliers(X, 'focalLength')
y = y[X.index]  # Update target after removing outliers

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Train an AdaBoost classifier
adboost = AdaBoostClassifier()
adboost.fit(X_train, y_train)

# Get the accuracy on the training and testing sets
training_accuracy = accuracy_score(y_train, adboost.predict(X_train))
testing_accuracy = accuracy_score(y_test, adboost.predict(X_test))

# Input data to predict
st.title("Tuna Sex Predictor App")
st.header("Predict Tuna Sex")

gonadal_weight = st.number_input("Enter gonadalWeight", min_value=X['gonadalWeight'].min(), max_value=X['gonadalWeight'].max(), value=X['gonadalWeight'].mean(), step=0.01)
focal_length = st.number_input("Enter focalLength", min_value=X['focalLength'].min(), max_value=X['focalLength'].max(), value=X['focalLength'].mean(), step=0.01)

if st.button("Click here to predict"):
    input_data = np.array([gonadal_weight, focal_length]).reshape(1, -1)
    prediction = adboost.predict(input_data)
    if prediction[0] == 0:
        st.write("The tuna is Female.")
    else:
        st.write("The tuna is Male.")

# Visualization dropdown menu
st.header("Data Visualization")
visualization_option = st.selectbox("Select Visualization:", ["None", "Box Plot", "Histogram", "Scatter Plot"])

# Set the Seaborn configuration to disable the warning
st.set_option('deprecation.showPyplotGlobalUse', False)

if visualization_option == "Box Plot":
    st.subheader("Box Plot for 'gonadalWeight'")
    sns.boxplot(data=X.assign(Sex=y), x='Sex', y='gonadalWeight')
    plt.xlabel('Sex')
    plt.ylabel('gonadalWeight')
    st.pyplot()

elif visualization_option == "Histogram":
    st.subheader("Histogram for 'focalLength'")
    sns.histplot(data=X.assign(Sex=y), x='focalLength', hue='Sex', kde=True)
    plt.xlabel('focalLength')
    st.pyplot()

elif visualization_option == "Scatter Plot":
    st.subheader("Scatter Plot for 'gonadalWeight' and 'focalLength'")
    sns.scatterplot(data=X.assign(Sex=y), x='gonadalWeight', y='focalLength', hue='Sex')
    plt.xlabel('gonadalWeight')
    plt.ylabel('focalLength')
    st.pyplot()

# Model Evaluation
st.header("Model Evaluation")
st.write("Accuracy on training data:", round(training_accuracy * 100), "%")
st.write("Accuracy on testing data:", round(testing_accuracy * 100), "%")

# Footer
footer = """<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Coyright @NgoshaJr</a></p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)