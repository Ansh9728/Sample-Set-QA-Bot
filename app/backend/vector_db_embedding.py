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
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

    base_embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        # model_kwargs={'device': device},
        encode_kwargs=encode_kwargs
    )

    return base_embedding_model

def create_documents_chunks(documents):
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
    def __init__(self, index_name, embedding_model) -> None:
        self.index_name = index_name
        self.embedding_model = embedding_model
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.index = None

    def pinecone_index(self):
        # if not self.pc.has_index(self.index_name):
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
                
            # self.index = self.pc.Index(self.index_name)
        index = self.pc.list_indexes().get('indexes')
        # print("Index name", index[0].get('name'))
        self.index = index[0].get('name')
        
        if self.index:
            print(f"Pinecone Index {self.index_name} Created Successfully")
        else:
            print(f"Pinecone Index {self.index_name} Creation Failed")

    def store_vector_embedding_to_pinecone(self, documents):

        # index = self.pinecone_index()
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

# urls = [
#     "https://en.wikipedia.org/wiki/Rajendra_Prasad",
#     "https://en.wikipedia.org/wiki/Premchand"
# ]

# loader = WebBaseLoader(urls)
# documents = loader.load()

# pinecone_index_name = 'smartset'
# model_name = "Snowflake/snowflake-arctic-embed-s"
# embedding_model = get_embedding_model(model_name)

# vc = VectorStore(pinecone_index_name, embedding_model)
# vector_db = vc.store_vector_embedding_to_pinecone(documents=documents)
# res = vector_db.similarity_search("Who is Rajender prasad")
# print(res)
# print(len(embedding_model.embed_query('hi')))