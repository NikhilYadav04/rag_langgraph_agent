import os
from dotenv import load_dotenv


load_dotenv()

# vector database
PINECONE_API_KEY= os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT= os.getenv("PINECONE_ENVIRONMENT","us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME","rag-index")

# Groq
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

# Tavily
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Embedding Model
EMBED_MODEL = os.getenv("EMBED_MODEL")

# Google API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")