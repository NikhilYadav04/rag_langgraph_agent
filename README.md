# 🤖 AI RAG Agent with LangGraph
<a href="https://youtu.be/mKupDsh9NjE?si=ah339pbRWZf9ciLr" title="YouTube Overview">
  <img src="https://img.icons8.com/ios-filled/20/FF0000/youtube-play.png" />
</a>

<img width="1840" height="853" alt="image" src="https://github.com/user-attachments/assets/d8a545c3-e7c3-47fc-9d39-05bbcddc2198" />

An intelligent Retrieval-Augmented Generation (RAG) agent built with LangGraph, FastAPI, and Streamlit. This agent intelligently routes queries between an internal knowledge base (vector store) and web search to provide accurate, context-aware responses.

## 🌟 Features

- **Intelligent Query Routing**: Automatically determines whether to use internal knowledge base, web search, or both
- **PDF Document Upload**: Upload and index PDF documents into the knowledge base
- **Web Search Integration**: Fallback to Tavily web search when internal knowledge is insufficient
- **Conversational Memory**: Maintains chat history across sessions using LangGraph checkpointing
- **Workflow Transparency**: Detailed trace of agent decision-making process
- **Toggle Web Search**: Enable/disable web search functionality on demand
- **Vector Database**: Powered by Pinecone for efficient similarity search
- **Modern UI**: Clean, interactive Streamlit frontend

## 🏗️ Architecture

### Backend (FastAPI)
- **LangGraph Agent**: Multi-node workflow graph with intelligent routing
- **Pinecone Vector Store**: Stores embedded document chunks
- **Groq LLM**: Fast inference with Llama 3.3 70B
- **Tavily Search**: Real-time web search capabilities

### Frontend (Streamlit)
- Interactive chat interface
- Document upload functionality
- Agent settings configuration
- Workflow trace visualization

### Agent Workflow

```
User Query → Router Node
              ├─→ RAG Lookup → Sufficiency Check
              │                 ├─→ Sufficient → Answer Node
              │                 └─→ Not Sufficient → Web Search Node → Answer Node
              ├─→ Web Search Node → Answer Node
              ├─→ Answer Node (Direct)
              └─→ End (Greetings)
```

## 📋 Prerequisites

- Python 3.10+
- Docker (optional, for containerized deployment)
- API Keys:
  - Pinecone API Key
  - Groq API Key
  - Tavily API Key
  - Google API Key (for embeddings)

## 🚀 Installation

### Local Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd rag-agent-project
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install backend dependencies**
```bash
cd backend
pip install -r requirements.txt
```

4. **Install frontend dependencies**
```bash
cd ../frontend
pip install -r requirements.txt
```

5. **Configure environment variables**

Create a `.env` file in the project root:

```env
# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=rag-index

# Groq Configuration
GROQ_API_KEY=your_groq_api_key

# Tavily Configuration
TAVILY_API_KEY=your_tavily_api_key

# Google API Configuration
GOOGLE_API_KEY=your_google_api_key
EMBED_MODEL=gemini-embedding-001

# Frontend Configuration
FASTAPI_BASE_URL=http://localhost:8000

# Railway Deployment (optional)
PORT=8000
```

### Docker Setup

1. **Build the Docker image**
```bash
docker build -t rag-agent .
```

2. **Run the container**
```bash
docker run -p 8000:8000 --env-file .env rag-agent
```

## 🎯 Usage

### Running Locally

1. **Start the FastAPI backend**
```bash
cd backend
uvicorn main:app --reload --port 8000
```

2. **Start the Streamlit frontend** (in a new terminal)
```bash
cd frontend
streamlit run app.py
```

3. **Access the application**
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Using the Application

1. **Upload Documents**
   - Navigate to the "Upload Document to Knowledge Base" section
   - Upload PDF files to build your knowledge base
   - Files are automatically processed and indexed

2. **Configure Settings**
   - Toggle "Enable Web Search" to control web search fallback
   - When disabled, agent relies solely on uploaded documents

3. **Chat with Agent**
   - Type your question in the chat input
   - View the agent's response and workflow trace
   - Expand "Agent Workflow Trace" to see decision-making process

## 📡 API Endpoints

### `/upload-document/` (POST)
Upload PDF documents to the knowledge base

**Request:**
- Form-data with PDF file

**Response:**
```json
{
  "message": "PDF 'example.pdf' successfully uploaded and indexed.",
  "filename": "example.pdf",
  "processed_chunks": 42
}
```

### `/chat/` (POST)
Chat with the RAG agent

**Request:**
```json
{
  "session_id": "unique-session-id",
  "query": "What is diabetes?",
  "enable_web_search": true
}
```

**Response:**
```json
{
  "response": "Diabetes is a chronic condition...",
  "trace_events": [
    {
      "step": 1,
      "node_name": "router",
      "description": "Router decided: 'rag'",
      "details": {},
      "event_type": "router_decision"
    }
  ]
}
```

### `/health` (GET)
Health check endpoint

### `/clear-index` (POST)
Clear all documents from the Pinecone index (Admin)

## 🧪 Agent Components

### Router Node
- Analyzes user query
- Decides optimal routing: RAG, Web Search, Direct Answer, or End
- Respects web search toggle setting

### RAG Lookup Node
- Retrieves relevant chunks from Pinecone vector store
- Uses LLM judge to assess sufficiency
- Routes to web search if insufficient

### Web Search Node
- Queries Tavily for current web information
- Returns top 3 most relevant results
- Can be disabled via settings

### Answer Node
- Synthesizes final response
- Combines RAG context and/or web results
- Uses Groq's Llama 3.3 70B for generation

## 🔧 Configuration

### Vector Store Configuration
- **Embedding Model**: Google Gemini Embedding (3072 dimensions)
- **Chunk Size**: 800 characters
- **Chunk Overlap**: 200 characters
- **Similarity Metric**: Cosine

### LLM Configuration
- **Model**: Llama 3.3 70B (via Groq)
- **Router Temperature**: 0 (deterministic)
- **Judge Temperature**: 0 (deterministic)
- **Answer Temperature**: 0.7 (creative)

## 🛠️ Technology Stack

- **Backend Framework**: FastAPI
- **Agent Framework**: LangGraph
- **LLM Provider**: Groq (Llama 3.3 70B)
- **Vector Database**: Pinecone
- **Embeddings**: Google Gemini
- **Web Search**: Tavily
- **Frontend**: Streamlit
- **Document Processing**: PDFPlumber
- **Deployment**: Docker, Railway

## 📊 Project Structure

```
rag-agent-project/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── agent.py             # LangGraph agent implementation
│   ├── vectorstore.py       # Pinecone vector store logic
│   ├── config.py            # Configuration management
│   └── requirements.txt     # Backend dependencies
├── frontend/
│   ├── app.py               # Streamlit main application
│   ├── ui_components.py     # UI rendering functions
│   ├── backend_api.py       # API client functions
│   ├── session_manager.py   # Session state management
│   ├── config.py            # Frontend configuration
│   └── requirements.txt     # Frontend dependencies
├── Dockerfile               # Docker configuration
├── .env                     # Environment variables (not in repo)
└── README.md               # This file
```

Built with ❤️ using LangGraph and FastAPI

