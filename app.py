import streamlit as st

# Create an iterator from a list of keys
keys = iter(['key1', 'key2', 'key3'])

# Use the next key from the iterator for the input box
key_for_input = next(keys)

# Create a single text input box with a dynamic key
user_input = st.text_input("Enter something:", key=key_for_input)

# Display the input value and the key used
st.write(f"You entered: {user_input}")
st.write(f"Using key: {key_for_input}")
