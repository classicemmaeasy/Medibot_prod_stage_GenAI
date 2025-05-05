from flask import Flask, render_template, jsonify, request
from src.helper import *
# from store_index import text_chunks
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from src.helper import download_hugging_face_embeddings

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


from langchain.vectorstores import Pinecone
# from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

from src.prompt import *


app= Flask(__name__)


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

GROQ_API_KEY=os.environ["GROQ_API_KEY"]


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


embeddings = download_hugging_face_embeddings()


index_name = "medicalchatbot"



llm=ChatGroq(model='llama-3.1-8b-instant') #mixtral-8x7b-32768
# llm.model_name, llm
 
# Embed each chunk and upsert the embedding into your database

docsearch= PineconeVectorStore.from_existing_index(
    index_name =index_name,
    embedding=embeddings,
)

retriever=docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})  

template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain= create_stuff_documents_chain(llm,template)
rag_chain= create_retrieval_chain(retriever, question_answer_chain)




@app.route('/')
def index():
    return render_template('chat.html')



@app.route('/get', methods=["GET","POST"])

def chat():
    msg= request.form["msg"]
    input=msg
    print(input)
    response = rag_chain.invoke({"input": input})
    print('Response: ', response['answer'])
    return str(response['answer'])



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug =True)