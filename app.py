import streamlit as st
from multiapp import MultiApp
from apps import home,model1,model2,model3,model4 # import your app modules here

app = MultiApp()

st.markdown("""
# Inteligencia de Negocios - Equipo B

""")
# Add all your application here
app.add_app("Home", home.app)
app.add_app("Modelo SVR", model1.app)
app.add_app("Modelo Logistic Regression", model2.app)
app.add_app("Modelo GRU", model3.app)
app.add_app("Modelo ARIMA", model4.app)

# The main app
app.run()
