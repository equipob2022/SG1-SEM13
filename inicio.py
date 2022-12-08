import streamlit as st

# Sharing variables among pages
from app import my_variable
from modelo1 import my_calc1

st.subheader("Inicio")
st.write(my_variable)

st.title(my_calc1)

