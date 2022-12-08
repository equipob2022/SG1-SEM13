import streamlit as st
from multiapp import MultiApp
from apps import home, data, model, description, SVR # import your app modules here

app = MultiApp()

st.markdown("""
# Inteligencia de Negocios - Equipo B

""")
# Add all your application here
app.add_app("Description", description.app)
app.add_app("Data", data.app)
app.add_app("Model", model.app)
app.add_app("SVR",SVR.app)
# The main app
app.run()
