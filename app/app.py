from backend.response_gen_model import chatbot_response_generation
import streamlit as st
from Data_Processing.pdf_ingestion import PdfIngestion
from backend.vector_db_embedding import VectorStore, get_embedding_model

def main():
    st.title("Sample Set QA Bot")

    # Handle file uploads
    uploaded_files = st.file_uploader(
        "Upload Your File",
        type=['pdf'],
        accept_multiple_files=True,
    )

    # Check if "Submit" button is clicked and files are uploaded
    if st.button("Submit") and uploaded_files:
        pdf_ingestion = PdfIngestion(uploaded_files)
        pdf_langchain_document = pdf_ingestion.load_pdf()

        # Flatten the list of documents
        flattened_list_doc = [element for sublist in pdf_langchain_document for element in sublist]

        # Setting up the VectorStore
        pinecone_index_name = 'smartset'
        model_name = "Snowflake/snowflake-arctic-embed-s"
        embedding_model = get_embedding_model(model_name)
        vector_store = VectorStore(pinecone_index_name, embedding_model=embedding_model)
        
        # Store the vector database in Streamlit session state
        st.session_state.vector_db = vector_store.store_vector_embedding_to_pinecone(flattened_list_doc)

    # Check if vector_db is in session state
    if 'vector_db' in st.session_state:
        question = st.text_input("Enter Your Question", key="question")

        if question:
            # Generate the response once a question is submitted
            retriever = st.session_state.vector_db.as_retriever()
            response = chatbot_response_generation(question=question, retriever=retriever)
            st.write("Chatbot Response:", response)
    else:
        st.info("Please upload a file and submit it to initialize the vector database.")

# Run the main function
if __name__ == "__main__":
    main()
