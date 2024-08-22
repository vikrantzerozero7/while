import streamlit as st
import time

# Generate a dynamic key based on the current time
while True:
  st.session_state.clear()
  key = 1
  query = st.text_input("Enter query", key =  key)
  key = key + 1

  st.write(query)
  st.session_state.clear()
