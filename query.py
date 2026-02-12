from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv

import os


# Load environment variables

load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

# nitializing embeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small",api_key=OPENAI_API_KEY)


# Initialize Pinecone

pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to existing Pinecone index

vector_store = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings
)

#Creating retriever

retriever = vector_store.as_retriever(search_kwargs={"k":3})


#initialize LLM

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY
)

print("✅ Ready to answer questions from Apple 2024 Annual Report")

#Ask questions in loop

while True:
    question = input("\nAsk a question (type 'exit or quit): ")

    if question.lower() in ["exit", "quit"]:
        break

    #Retrieve relevant chunks
    docs = retriever.invoke(question)

     # Combine retrieved text
    context = "\n\n".join(doc.page_content for doc in docs)

    # Prompt
    prompt = f"""
    You are a financial analyst assistant.

    Use ONLY the provided context to answer the question.
    If the answer is not found in the context, say:
    "I don't know based on the provided document."

    Provide a clear and concise answer.
    If numbers are mentioned, include exact figures.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    # Get response
    response = llm.invoke(prompt)

    print("\n🧠 Answer:\n", response.content)