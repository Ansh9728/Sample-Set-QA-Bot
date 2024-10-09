import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
load_dotenv()


pinecone_api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)