import streamlit as st

# Sharing variables among pages
from app import my_variable

st.subheader("Inicio")
st.write(my_variable)
