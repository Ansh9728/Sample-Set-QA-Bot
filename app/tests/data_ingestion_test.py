# from backend.vector_db_embedding import get_embedding_model, VectorStore
# from langchain_community.document_loaders import WebBaseLoader

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