from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
from langchain_pinecone import PineconeVectorStore


from dotenv import load_dotenv

load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY




extracted_data = load_pdf(data="Data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicalchatbot"


pc.create_index(
    name=index_name,
    dimension=384,  # Update this to match the embedding dimension
    metric="cosine",  # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)


# Embed each chunk and upsert the embedding into your database

docsearch= PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name =index_name,
    embedding=embeddings,
)