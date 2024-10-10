import os
import time
import uuid
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_pinecone import Pinecone as lang_pinecone
# from langchain.vectorstores import Pinecone as pn
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
# from pinecone import Pinecone, ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
load_dotenv()


def get_embedding_model(model_name="Snowflake/snowflake-arctic-embed-s"):
    # model_name = "Snowflake/snowflake-arctic-embed-l"
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


# pinecone_api_key = os.getenv("PINECONE_API_KEY")
# pc = Pinecone(api_key=pinecone_api_key)
# pinecone_index_name = "smart_set_assignment"
# embedding_model = get_embedding_model(model_name)
# res = embedding_model.embed_query("Hi how are you")
# print(res)


# def get_pinecone_index(index_name):
#     try:
#         if not pc.has_index(index_name):
#             pc.create_index(
#                 name=index_name,
#                 dimension=2,
#                 metric="cosine",
#                 spec=ServerlessSpec(
#                     cloud='aws', 
#                     region='us-east-1'
#                 ) 
#             )

#             while not pc.describe_index(index_name).status['ready']:
#                 time.sleep(1)
                
#             index = pc.Index(index_name)
        
#         return True
#     except Exception as e:
#         print(f"Error While Creating the Pinecone Index : {e}")

#     return False


# def store_vector_embedding_to_pinecone()

# class VectorStore():

#     def __init__(self, index_name, embedding_model) -> None:
#         self.index_name = index_name
#         self.embedding_model = embedding_model

#     def pinecone_index(self):

#         index = get_pinecone_index(self.index_name)
#         if index:
#             print(f"Pinecone Index {self.index_name} Created Successfully")
#         else:
#             print(f"Pinecone Index {self.index_name} Creation Failed")

#     def store_vector_embedding_to_pinecone(self, documents):
#         try:
            
#             documents_Chunk =create_documents_chunks(documents=documents)

#             lang_pinecone.from_documents(documents_Chunk, self.embedding_model, self.index_name)
#         except Exception as e:
#             print(f"Error While Storing Vector Embeddings to Pinecone Index : {e}")



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

            # vector_store = PineconeVectorStore(self.index, self.embedding_model)

            vector_db = PineconeVectorStore.from_documents(
                documents_Chunk,
                index_name = self.index_name,
                embedding=self.embedding_model
            )
            # lang_pinecone.from_documents(documents_Chunk, self.embedding_model, self.index)
            # embeddings = self.embedding_model.embed_documents(documents)
            # self.index.upsert(vectors=embeddings, ids=[f"doc-{i}" for i in range(len(documents))])
            # print(embeddings)
            print('Data stored succusfully')
            return vector_db

        print("Cannot store vector embeddings.")
        
        return None

urls = [
    "https://en.wikipedia.org/wiki/Rajendra_Prasad",
    "https://en.wikipedia.org/wiki/Premchand"
]

loader = WebBaseLoader(urls)
documents = loader.load()

pinecone_index_name = 'smartset'
model_name = "Snowflake/snowflake-arctic-embed-s"
embedding_model = get_embedding_model(model_name)

vc = VectorStore(pinecone_index_name, embedding_model)
vc.store_vector_embedding_to_pinecone(documents=documents)

# print(len(embedding_model.embed_query('hi')))