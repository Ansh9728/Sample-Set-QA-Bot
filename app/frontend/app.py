
import streamlit as st
from data_processing.pdf_ingestion import PdfIngestion
# from  data_processing.pdf_ingestion import DataIngestion

def main():

    st.title("Sample Set QA Bot")

    uploaded_files = st.file_uploader(
        "Upload Your File",
        type=['pdf'],
        accept_multiple_files=True,
    )

    if st.button("Submit"):
        pdf_ingestion = PdfIngestion(uploaded_files)
        pdf_content = pdf_ingestion.load_pdf()

        st.write(pdf_content)


main()