import gradio as gr
import logging
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from azure.search.documents import IndexDocumentsBatch
from azure.core.credentials import AzureKeyCredential
from openai import OpenAI

from python_files.langchain_openai_embeddings import get_retriever_with_chat_history

from python_files.constants import (
    AZURE_SEARCH_INDEX_NAME,
    APP_TITLE,
    APP_VERSION,
    GRADIO_SERVER_PORT,
    AZURE_STORAGE_URL,
    AZURE_STORAGE_CONTAINER_NAME,
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_ADMIN_KEY,
    OPENAI_API_KEY
)

from python_files.langchain_openai_embeddings import get_retriever
from python_files.document_ingestion_scheduler import process_messages_from_blob_storage

from tenacity import retry, stop_after_attempt, wait_fixed

# Logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(funcName)s - %(lineno)d - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S"
)

logging.debug("Starting doc_ui.py script")

client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize the retriever (This uses Azure Search to retrieve documents with Azure OpenAI Embeddings)
retriever = get_retriever(index_name=AZURE_SEARCH_INDEX_NAME)

# Updated function to generate responses using OpenAI's API
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def respond(message, chat_history):
    logging.debug(f"Received message: {message}")
    
    try:
        if isinstance(message, dict):
            message = message.get("input", "")
        message = str(message)
        logging.debug(f"Message type after conversion: {type(message)}")
        
        # Log the chat history passed in
        formatted_chat_history = [str(entry) for entry in chat_history]
        logging.debug(f"Formatted chat history: {formatted_chat_history}")

        retriever_chain = get_retriever_with_chat_history(index_name=AZURE_SEARCH_INDEX_NAME)
        logging.debug("Retriever chain created for context retrieval.")

        # Log message and chat history before context retrieval
        logging.debug(f"Retrieving context with message: {message} and chat history.")
        retrieved_context = retriever_chain.invoke({
            "input": message,
            "chat_history": formatted_chat_history
        })

        # Add logging for the retrieved context and its type
        logging.debug(f"Retrieved context type: {type(retrieved_context)}")
        logging.debug(f"Retrieved context content: {retrieved_context}")

        # Ensure retrieved context is structured properly
        context = retrieved_context.get('context', [])
        if not isinstance(context, list):
            raise ValueError("Context should be a list of retrieved documents")
        logging.debug(f"Context retrieved: {context}")

        # If no context found
        if not context:
            llm_message = "No relevant documents were found to answer this question."
            chat_history.append((message, llm_message))
            logging.debug("No relevant context. Response sent to user.")
            return "", chat_history

        # Include the content of each document in the context
        context_text = "\n\n".join([
            f"{doc.metadata['source']} (Page {doc.metadata.get('page', 'N/A')}):\n{doc.page_content}" 
            for doc in context if 'source' in doc.metadata
        ])
        logging.debug(f"Formatted context with content: {context_text}")

        if not context_text:
            context_text = "No relevant context found from documents."

        # Create LLM messages based on chat history and context
        messages = [
            {
                "role": "system", 
                "content": """
You are a helpful assistant. Answer based strictly on the provided document context. 

- **If the user asks an informational question based on the documents:**

- Provide a clear and concise answer.
- Include a `Reference:` section listing the sources and page numbers.

- **If the user input is a casual message or not related to the documents:**

- Respond naturally without including any references.

**Format your response as follows when references are applicable:**

[Response]

Reference: [source1] [page_number1], [source2] [page_number2], ...

Example:

The UNOS (United Network for Organ Sharing) computer system generates the ranked list of transplant candidates who are suitable to receive each organ.

Reference: test_doc.pdf Page 3, another_doc.pdf Page 1

If there is no context provided or the context is not relevant to the question, respond naturally without references.
"""
            },
            {"role": "system", "content": f"Context: {context_text}"}
        ]
        for user_message, assistant_message in chat_history:
            messages.append({"role": "user", "content": str(user_message)})
            messages.append({"role": "assistant", "content": str(assistant_message)})

        messages.append({"role": "user", "content": message})

        logging.debug(f"Messages passed to LLM: {messages}")

        # Sending the messages to the language model and logging the response
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=messages,
            temperature=0.0
        )

        logging.debug(f"Response structure: {response}")

        llm_message = response.choices[0].message.content.strip()
        logging.debug(f"LLM Response: {llm_message}")

        chat_history.append((message, llm_message))
        return "", chat_history

    except Exception as e:
        logging.error(f"Error in respond function: {e}", exc_info=True)
        return "", chat_history

# Function to update document count from Azure Blob Storage
async def update_document_count():
    logging.info("Updating the document count.")
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_URL)
    container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER_NAME)
    logging.debug("Checking the document count in Azure Blob Storage...")
    blob_list = container_client.list_blobs()
    count = sum(1 for _ in blob_list)  # Count the number of blobs
    doc_count_text = f"Number of documents in Blob Storage: {count}"
    logging.info(doc_count_text)
    return doc_count_text

# Function to start the vectorization process (document ingestion and vector indexing)
async def start_vectorization():
    await process_messages_from_blob_storage()

# Function to clear the Azure Search index
def clear_database():
    logging.info("Clear Database button pressed")

    # Create a search client
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
    )

    try:
        # Retrieve all document IDs (assuming 'id' is the key field)
        results = search_client.search(search_text="*", select=["id"])
        document_ids = [{"id": doc["id"]} for doc in results]

        if document_ids:
            # Create a batch delete operation
            batch = IndexDocumentsBatch()
            batch.add_delete_actions(*document_ids)

            # Submit the batch operation to delete the documents
            search_client.index_documents(batch=batch)
            logging.info(f"Deleted {len(document_ids)} documents from the index.")
        else:
            logging.info("No documents found in the index.")

    except Exception as e:
        logging.error(f"Failed to clear database: {e}", exc_info=True)

def create_gradio_interface():
    with gr.Blocks(css="""
        /* Make the entire Blocks container take full viewport height */
        .gradio-container {
            height: 100vh;
            display: flex;
            flex-direction: column;
        }

        /* Make the main content area grow to fill available space */
        .main-content {
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            padding: 10px; /* Optional: Add some padding */
        }

        /* Make the chat area grow and be scrollable */
        .chat-area {
            flex-grow: 1;
            overflow-y: auto;
            margin-bottom: 10px; /* Optional: Space between chat and textbox */
        }

        /* Style the textbox to be fixed at the bottom */
        .input-box {
            width: 100%;
        }
    """) as demo:
        # Header Section
        gr.Markdown(f"""
        # {APP_TITLE}
        ### (Version: {APP_VERSION})
        """)
        
        # Control Buttons Row
        with gr.Row():
            gr.Markdown('''
            <style>
            a:link { color: #0000EE; } /* Unvisited link */
            a:visited { color: #551A8B; } /* Visited link */
            a:active { color: #FF0000; } /* Active link */
            </style>
            <a href="https://cmu.app.box.com/folder/282324107536" style="font-size:20px; font-weight:bold;">
                Box Folder
            </a>
            ''')
            gr.Button("Start Vectorization", variant="primary").click(start_vectorization)
            gr.Button("Clear Database", variant="secondary").click(clear_database)

        # Document Count Display
        doc_count_element = gr.Markdown("Number of documents in Blob Storage: Loading...")

        # Periodically Update Document Count
        demo.load(
            update_document_count, 
            inputs=[], 
            outputs=doc_count_element, 
            every=25  # Update every 25 seconds
        )

        # Q&A Tab with Full-Screen Chatbot
        with gr.Tab("Q&A"):
            with gr.Column(elem_id="main-content", elem_classes="main-content"):
                with gr.Column(elem_classes="chat-area"):
                    chat = gr.Chatbot(label="AI Response")
                with gr.Column(elem_classes="input-box"):
                    msg = gr.Textbox(
                        label="Question", placeholder="Input your query here and then press Enter"
                    )
                    msg.submit(respond, [msg, chat], [msg, chat])

        return demo

# Launch the Gradio demo
demo = create_gradio_interface()

# Configure the Gradio server port
port = GRADIO_SERVER_PORT
logging.debug(f"Configured port number - {port}")