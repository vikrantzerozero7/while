import streamlit as st
import fitz  # PyMuPDF

# File uploader to select multiple PDF files
uploaded_files = st.sidebar.file_uploader("Choose a file", accept_multiple_files=True, key="fileUploader", type="pdf")

# Initialize an empty list to store the opened PDF documents
pdf_d = []

if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        df = fitz.open(stream=uploaded_file.read(), filetype="pdf")  # Open the uploaded PDF file
        pdf_d.append(df)  # Add the opened PDF document to the list

# Now pdf_d contains all the opened PDF documents
st.write(f"{len(pdf_d)} PDF(s) have been uploaded and opened.")
