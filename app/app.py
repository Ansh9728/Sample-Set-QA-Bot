
import streamlit as st
from Data_Processing.pdf_ingestion import PdfIngestion
# from app.backend.vector_db_embedding import VectorStore, get_embedding_model
# from app.backend.vector_db_embedding import VectorStore, get_embedding_model
# from .backend.vector_db_embedding import VectorStore, get_embedding_model
# from  data_processing.pdf_ingestion import DataIngestion
from backend.vector_db_embedding import  VectorStore, get_embedding_model

def main():

    st.title("Sample Set QA Bot")

    uploaded_files = st.file_uploader(
        "Upload Your File",
        type=['pdf'],
        accept_multiple_files=True,
    )

    if st.button("Submit") and uploaded_files:
        pdf_ingestion = PdfIngestion(uploaded_files)
        pdf_langchain_document = pdf_ingestion.load_pdf()

        # Display the PDF content (you may want to format this differently)
        # print(type(pdf_langchain_document))

        pinecone_index_name = 'smartset'
        model_name = "Snowflake/snowflake-arctic-embed-s"
        embedding_model = get_embedding_model(model_name)

        vector_store = VectorStore(pinecone_index_name, embedding_model=embedding_model)


        flattened_list_doc = [element for sublist in pdf_langchain_document for element in sublist]
        print(flattened_list_doc)

        
        print("length of docs",len(flattened_list_doc))
        print('\n')

        # for pdf_doc in pdf_langchain_document:
        #     vector_store.store_vector_embedding_to_pinecone(pdf_doc)

        # print(pdf_langchain_document[0])

        vector_db = vector_store.store_vector_embedding_to_pinecone(flattened_list_doc)
        print("Performed similarity",vector_db.similarity_search('Who is Anshu'))
        st.write(flattened_list_doc)

if __name__=="__main__":

    main()