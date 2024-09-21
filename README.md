# IntelliDoc AI

A **Retrieval-Augmented Generation (RAG)** application leveraging Azure OpenAI, LangChain, and Gradio to facilitate intelligent querying of scientific documents.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Running the Application](#running-the-application)
- [Logging](#logging)
- [Contributing](#contributing)
- [Contact](#contact)

## Overview

The RAG-LLM project is designed to enable users to query scientific documents using advanced language models. By integrating Azure OpenAI services with LangChain and Gradio, the application provides a robust interface for retrieving and generating responses based on the content of uploaded documents.

## Features

- **Document Ingestion:** Automatically processes and indexes PDF documents stored in Azure Blob Storage.
- **Intelligent Retrieval:** Utilizes Azure Search and OpenAI embeddings to fetch relevant document sections based on user queries.
- **Interactive UI:** Built with Gradio, offering a user-friendly chat interface for seamless interaction.
- **Scheduled Processing:** Employs APScheduler to periodically ingest and process new documents.
- **Clear Database Functionality:** Allows users to reset the search index when needed.
- **Comprehensive Logging:** Detailed logs for monitoring and debugging purposes.

## Technologies Used

- **Python 3.11**
- **Azure OpenAI Services**
- **LangChain**
- **Gradio**
- **FastAPI**
- **Azure Blob Storage**
- **Azure Search**
- **Azure Storage Queues**
- **APScheduler**
- **Tenacity** (for retry mechanisms)
- **pdfplumber** (for PDF processing)

### Key Components

- **constants.py:** Manages configuration and environment variables.
- **doc_ui.py:** Defines the Gradio user interface for interacting with the model.
- **document_ingestion_scheduler.py:** Handles the ingestion and processing of documents from Azure Blob Storage.
- **langchain_llm_utils.py:** Utility functions for interacting with LangChain and OpenAI.
- **langchain_openai_embeddings.py:** Sets up the embeddings and retrievers using LangChain.
- **server.py:** Sets up the FastAPI server and integrates the Gradio interface.
- **version_router.py:** Provides API routes for version information.

## Prerequisites

- **Python 3.11** or higher
- **Azure Account** with access to:
  - Azure OpenAI
  - Azure Blob Storage
  - Azure Search
  - Azure Storage Queues
- **Git** for version control

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. **Environment Variables**

   Create a `.env` file in the root directory with the following variables:

   ```env
   # Gradio Server
   GRADIO_SERVER_PORT=8888

   # Azure OpenAI
   AZURE_OPENAI_MODEL_NAME=your-model-name
   AZURE_OPENAI_API_ENDPOINT=https://your-endpoint.openai.azure.com/
   AZURE_OPENAI_API_KEY=your-azure-openai-api-key
   AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
   AZURE_OPENAI_API_VERSION=2024-08-01-preview
   AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME=your-embeddings-deployment-name
   AZURE_OPENAI_EMBEDDINGS_MODEL_NAME=your-embeddings-model-name
   AZURE_OPENAI_API_EMBEDDINGS_VERSION=2024-02-01

   # OpenAI API
   OPENAI_API_KEY=your-openai-api-key
   OPENAI_COMPLETION_MODEL=gpt-4o-mini-2024-07-18
   OPENAI_EMBEDDING_MODEL=text-embedding-3-large

   # Azure Search
   AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net/
   AZURE_SEARCH_ADMIN_KEY=your-search-admin-key
   AZURE_SEARCH_INDEX_NAME=your-index-name

   # Azure Storage
   AZURE_STORAGE_URL=https://your-storage-account.blob.core.windows.net/
   AZURE_STORAGE_CONTAINER_NAME=your-container-name
   AZURE_STORAGE_QUEUE_NAME=your-queue-name

   # Application Settings
   LLM_TYPE=Azure
   MESSAGES_PER_PAGE=10
   MAX_MESSAGES_PER_POLL=50
   DELETE_BLOBS=False
   CRON_EXPRESSION="0 0/2 * * *"
   APP_TITLE="Your App Title"
   APP_VERSION="1.0.0"
   MAX_CHAT_HISTORY_LENGTH=5
   ```

   **Note:** Replace the placeholder values (`your-...`) with your actual Azure service credentials and desired configurations.

2. **Azure Services Setup**

   - **Azure Blob Storage:** Store your PDF documents in the specified container.
   - **Azure Search:** Create an index matching the configuration in `constants.py`.
   - **Azure Storage Queues:** Set up a queue to manage document ingestion tasks.

## Usage

1. **Prepare Your Documents**

   - Place your PDF documents in the `data/` directory or upload them to the configured Azure Blob Storage container.

2. **Run the Server**

   ```bash
   python python_files/server.py
   ```
   OR

   ```bash
   uvicorn - uvicorn server:app --host 0.0.0.0 --port 8000 --reload
   ```

   This will start the FastAPI server and launch the Gradio interface.

3. **Access the Application**

   - Open your browser and navigate to `http://localhost:8888/` to interact with the Gradio UI.
   - For API interactions, use the FastAPI endpoints (e.g., `http://localhost:8000/ask`).

## Running the Application

The application integrates both FastAPI and Gradio. Here's how to run and interact with both:

1. **Start the Server**

   ```bash
   python python_files/server.py
   ```

   - **FastAPI Server:** Runs on `http://0.0.0.0:8000/`
   - **Gradio Interface:** Accessible at `http://localhost:8888/`

2. **Using the Gradio UI**

   - **Q&A Tab:** Enter your queries related to the ingested documents and receive intelligent responses.
   - **Control Buttons:**
     - **Start Vectorization:** Initiates the document ingestion and vectorization process.
     - **Clear Database:** Clears the Azure Search index.

3. **API Endpoints**

   - **Health Check:** `GET /`  
     Returns a welcome message.

   - **Ask:** `POST /ask`  
     Submit a prompt and receive an AI-generated response based on the document context.

     **Request Body:**

     ```json
     {
       "prompt": "Your question here",
       "chat_history": [
         {"User: Previous question": "AI: Previous answer"}
       ]
     }
     ```

     **Response:**

     ```json
     {
       "answer": "AI-generated response based on documents."
     }
     ```

## Logging

The application employs comprehensive logging for monitoring and debugging:

- **Console Logging:** Logs are output to the console for real-time monitoring.

Log messages include details about:

- Server startup and shutdown
- Document ingestion processes
- API requests and responses
- Error handling and exceptions

## Contributing

As this is a private repository, contributions are managed internally. Please follow the standard Git workflow:

1. **Fork the Repository**
2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add your message"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request**

## Contact

For any questions or support, please contact [rahul31pujari@gmail.com](mailto:rahul31pujari@gmail.com).

---

*Developed by Rahul Pujari*