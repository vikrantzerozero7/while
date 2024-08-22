# app.py
import module
import streamlit as st
from module import x  # Importing x from module.py

if x == 3:
    st.write("zone")
else:
    st.write("x is not 3")
