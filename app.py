import streamlit as st

my_variable = "From Main App.py Page"

def main():
    st.title("Streamlit Multi-Page")
    st.subheader("Main Page")
    st.write(my_variable)
    
    submenu = st.sidebar.selectbox("SubMenu",["Inicio","Modelo 1","Modelo 2","Modelo 3","Modelo 4"])
    if submenu == "Inicio":
       st.subheader("Inicio")
    
    
if __name__ == '__main__':
  main()
