from backend.response_gen_model import chatbot_response_generation
import streamlit as st
from Data_Processing.pdf_ingestion import PdfIngestion
from backend.vector_db_embedding import VectorStore, get_embedding_model
from langgraph.errors import GraphRecursionError



# Frontend and Integration with Backend

def main():
    st.title("Sample Set QA Bot")

    # Handle file uploads
    uploaded_files = st.file_uploader(
        "Upload Your File",
        # type=['pdf'],
        type=['pdf', 'csv', 'txt'],
        accept_multiple_files=True,
    )

    # Check if "Submit" button is clicked and files are uploaded
    if st.button("Submit") and uploaded_files:
        # pdf_ingestion = PdfIngestion(uploaded_files)
        # pdf_langchain_document = pdf_ingestion.load_pdf()

        ingestion = FileIngestion(uploaded_files)
        pdf_langchain_document = ingestion.load_files()

        # Flatten the list of documents
        flattened_list_doc = [element for sublist in pdf_langchain_document for element in sublist]

        # Setting up the VectorStore
        pinecone_index_name = 'smartset'
        model_name = "Snowflake/snowflake-arctic-embed-s"
        embedding_model = get_embedding_model(model_name)
        vector_store = VectorStore(pinecone_index_name, embedding_model=embedding_model)
        
        # Store the vector database in Streamlit session state
        st.session_state.vector_db = vector_store.store_vector_embedding_to_pinecone(flattened_list_doc)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    if 'vector_db' in st.session_state:
        retriever = st.session_state.vector_db.as_retriever()

        if question := st.chat_input("Enter Your Question?"):

            st.session_state.messages.append({"role": "user", "content": question})

            try:
                response, retrieved_documents = chatbot_response_generation(question=question, retriever=retriever)

                with st.chat_message("assistant"):
                    st.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})


                # Display the retrieved documents in the sidebar
                st.sidebar.title("Retrieved Documents")
                for doc in retrieved_documents:
                    # You can choose to display document metadata or a snippet of the document content
                    # st.sidebar.markdown(f"**Content:** {doc.page_content}...")  # Show first 200 characters
                    st.sidebar.markdown(f"**Content:** {doc}...")
                    st.sidebar.markdown("---")

            except GraphRecursionError as e:
                with st.chat_message("assistant"):
                    st.markdown(f"Error Not Related to docs ")
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": f"Error Not Related to docs "})
                # st.error("Error: Graph recursion limit reached. Please try again later or contact support.")

# Run the main function
if __name__ == "__main__":
    main()
