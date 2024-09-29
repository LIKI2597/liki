import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.linear_model import LinearRegression

# title
st.title("Webapp using Streamlit")

# image
st.image('streamlit.png', width=300)

st.title("Case Study on Diamond Dataset")

# load dataset
data = sns.load_dataset("diamonds")
st.write("Shape of the dataset:", data.shape)

# sidebar menu
menu = st.sidebar.radio("Menu", ["Home", "Prediction Price"])

# Home Page Content
if menu == "Home":
    st.image("download.png", width=350)

    # display tabular data
    st.header("Tabular Data of Diamond")
    if st.checkbox("Tabular Data"):
        st.table(data.head(150))

    # statistical summary
    st.header("Statistical Summary of a DataFrame")
    if st.checkbox("Statistics"):
        st.table(data.describe())

    # correlation graph
    st.header("Correlation Graph")
    # Select only numeric columns for correlation
    numeric_data = data.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(10, 8))  # Adjusted figsize for better display
    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax)
    # Display heatmap in streamlit
    st.pyplot(fig)

    # Graphs
    st.title("Graphs")
    graph = st.selectbox("Different types of Graphs", ["Scatter plot", "Bar Graph", "Histogram"])

    # Scatter plot graph
    if graph == "Scatter plot":
        value = st.slider("Filter data using carat", 0, 6)
        filtered_data = data.loc[data["carat"] >= value]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=filtered_data, x="carat", y="price", hue="cut")
        st.pyplot(fig)

    # Bar graph
    if graph == "Bar Graph":
        fig, ax = plt.subplots(figsize=(3, 2))
        sns.barplot(x="cut", y=data.index, data=data)
        st.pyplot(fig)

    # Histogram graph
    if graph == "Histogram":
        fig, ax = plt.subplots(figsize=(3, 2))
        sns.histplot(data.price, kde=True)
        st.pyplot(fig)

# Prediction Price Page Content
if menu == "Prediction Price":
    st.title("Prediction Price of a Diamond")

    # Linear regression model
    lr = LinearRegression()

    # Independent variable (X) and dependent variable (y)
    X = np.array(data["carat"]).reshape(-1, 1)
    y = np.array(data["price"]).reshape(-1, 1)

    lr.fit(X, y)

    # Input for carat value
    value = st.number_input("Carat", 0.20, 5.01, step=0.15)
    value = np.array(value).reshape(-1, 1)

    # Prediction
    prediction = lr.predict(value)[0]

    # Display prediction
    if st.button("Price Prediction in ($)"):
        st.write(f"Predicted Price: ${prediction}")
