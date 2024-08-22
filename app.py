import streamlit as st

def process_input(user_input):
    # Process the input (replace this with your logic)
    return f"Processed: {user_input}"

if 'stop' not in st.session_state:
    st.session_state.stop = False

# Continuously ask for input until 'stop' button is pressed
while not st.session_state.stop:
    st.experimental_rerun()
    user_input = st.text_input("Enter something:")
    if user_input:
        result = process_input(user_input)
        st.write(result)
    
    if st.button("Stop"):
        st.session_state.stop = True
        st.write("Loop stopped.")
    
    st.write("Waiting for the next input...")
