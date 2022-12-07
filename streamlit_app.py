import streamlit as st
from multiapp import MultiApp
from apps import home, model, model3 , modelRF  # import your app modules here model2

app = MultiApp()

st.markdown("""
#  Inteligencia de Negocios - Equipo B
""")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Modelo LSTM", modelo1.app)
# The main app
app.run()
