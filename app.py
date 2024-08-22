import streamlit as st

# Define the number of iterations
num_iterations = 5

# Loop through the specified number of iterations
for i in range(num_iterations):
    st.write(f"Iteration {i + 1}")
    
    # Display a text input box
    user_input = st.text_input(f"Enter something for iteration {i + 1}")
    
    # Check if the user has entered something
    if user_input:
        st.write(f"You entered: {user_input}")
    
    # Add a button to proceed to the next iteration
    if st.button(f"Next {i + 1}"):
        continue
