import streamlit as st
import time

# Generate a dynamic key based on the current time
while True:

  query = st.text_input("Enter query")

  st.write(query)
  st.session_state.clear()
