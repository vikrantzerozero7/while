# app.py
import module
import streamlit as st
from module import x  # Importing x from module.py

if x == 3:
    x= st.text_input("enter")
    st.write(x)
else:
    st.write("x is not 3")
