import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")


# Load PDF

loader = PyPDFLoader(".\data\Apple Inc 2024 Annual Report.pdf")

documents = loader.load()

print("Total documents:",len(documents))

# print(documents[27].page_content[:500])

# Chunk PDF

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap=100
)

docs = text_splitter.split_documents(documents=documents)


print("Total doucments after chunking", len(docs))



# embedding

embeddings = OpenAIEmbeddings(model="text-embedding-3-small",api_key=OPENAI_API_KEY)

#nitialize pinecone

pc = Pinecone(api_key=PINECONE_API_KEY)


# creating index in pinecone if not exist

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )


    print("Index has been created")



# Create vector store and upload documents
vector_store = PineconeVectorStore.from_documents(
    documents=docs,
    embedding=embeddings,
    index_name=INDEX_NAME
    
    
)

print("Documents successfully uploaded to Pinecone")