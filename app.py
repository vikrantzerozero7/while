import streamlit as st
import time
x = 9 
st.write(f"text_input_{int(time.time())}")
# Generate a dynamic key based on the current time
if x == 9: 
  
  query = st.text_input("Enter query")
  

  st.write(query)
  st.session_state.clear()
