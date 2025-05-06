import streamlit as st

st.title("My Streamlit App from Jupyter")

name = st.text_input("Enter your name:")

if name:
    st.write(f"Hello, {name}!")

number = st.slider("Select a number:", 0, 100)
st.write(f"You selected: {number}")
