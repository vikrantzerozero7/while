import streamlit as st
import time

# Generate a dynamic key based on the current time
dynamic_key = f"text_input_{int(time.time())}"

query = st.text_input("Enter query", key=dynamic_key)
time.sleep(3)
