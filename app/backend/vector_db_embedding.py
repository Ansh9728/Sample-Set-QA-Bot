import os
import time
import uuid
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
# from pinecone import Pinecone, ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
load_dotenv()


def get_embedding_model(model_name="Snowflake/snowflake-arctic-embed-s"):
    """
        Returns an embedding model for encoding text into vector embeddings using HuggingFace models.

        Args:
            model_name (str, optional): The name of the HuggingFace model to be used for generating embeddings. 
            
            Defaults to "Snowflake/snowflake-arctic-embed-s".

        Returns:
            HuggingFaceEmbeddings: An embedding model with normalized embeddings enabled for cosine similarity computation.
        
    
    """
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

    base_embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        # model_kwargs={'device': device},
        encode_kwargs=encode_kwargs
    )

    return base_embedding_model


def create_documents_chunks(documents):

    """
        Splits a list of documents into smaller chunks using a Recursive Character Text Splitter.

        Args:
            documents (list): A list of Langchain Document objects to be split into smaller chunks.
        
        Returns:
            list: A list of smaller document chunks generated from the input documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    docs = text_splitter.split_documents(documents)
    return docs


class VectorStore():

    """
    A class to handle the creation and management of a Pinecone vector index, 
    as well as storing document embeddings into the Pinecone vector database.

    Args:
        index_name (str): The name of the Pinecone index to be created or used.
        embedding_model (HuggingFaceEmbeddings): The embedding model used for generating document embeddings.
    
    Attributes:
        index_name (str): The name of the Pinecone index.
        embedding_model (HuggingFaceEmbeddings): The embedding model used for generating embeddings.
        pc (Pinecone): An instance of the Pinecone class for interacting with the Pinecone service.
        index (str or None): The name of the created or retrieved Pinecone index.
    """

    def __init__(self, index_name, embedding_model) -> None:

        """
        Initializes the VectorStore class with the specified Pinecone index name 
        and embedding model.

        Args:
            index_name (str): The name of the Pinecone index to be created or used.
            embedding_model (HuggingFaceEmbeddings): The embedding model to be used for creating embeddings.
        """

        self.index_name = index_name
        self.embedding_model = embedding_model
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.index = None


    def pinecone_index(self):

        """
        Creates a Pinecone index if it does not exist, using the specified embedding model's 
        output dimension for index configuration.

        - Uses cosine similarity as the distance metric for the index.
        - If the index is successfully created or already exists, its name is stored in self.index.
        - If the index creation fails, a message is printed.

        Returns:
            None
        """
        if not self.pc.list_indexes().get('indexes'):
            self.pc.create_index(
                name=self.index_name,
                # dimension=len(self.embedding_model.embed_query("Hi")).
                # dimension=384,
                dimension=len(self.embedding_model.embed_query('hi')),
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws', 
                    region='us-east-1'
                ) 
            )

            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
                
        index = self.pc.list_indexes().get('indexes')
        self.index = index[0].get('name')
        
        if self.index:
            print(f"Pinecone Index {self.index_name} Created Successfully")
        else:
            print(f"Pinecone Index {self.index_name} Creation Failed")


    def store_vector_embedding_to_pinecone(self, documents):
        """
            Stores the document embeddings in the Pinecone index.

            Args:
                documents (list): A list of Langchain Document objects to be stored in the Pinecone vector database.

            Returns:
                PineconeVectorStore or None: If successful, returns the PineconeVectorStore object containing 
                the vector embeddings; otherwise, returns None.
        """

        self.pinecone_index()
        print(" Data Storinig...   Pinecone Index Name",self.index)
        if self.index:
              
            documents_Chunk =create_documents_chunks(documents=documents)

            vector_db = PineconeVectorStore.from_documents(
                documents_Chunk,
                index_name = self.index_name,
                embedding=self.embedding_model
            )
           
            print('Data stored succusfully')
            return vector_db

        print("Cannot store vector embeddings.")
        
        return None