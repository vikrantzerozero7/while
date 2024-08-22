import streamlit as st
while True:
    st.session_state.key = None
    query = st.text_input("enter query")
    st.write(query)
    query = None
    st.session_state.key = None
