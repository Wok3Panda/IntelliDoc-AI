import pdfplumber
import io
from typing import List, Iterator, Tuple
import asyncio
import base64
import json
import urllib.parse
from uuid import uuid4

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from langchain_community.vectorstores.azuresearch import AzureSearch
from azure.storage.queue import QueueClient, QueueMessage
from azure.storage.blob import BlobServiceClient, BlobClient
from langchain_core.documents import Document

from python_files.constants import (
    AZURE_STORAGE_URL,
    AZURE_STORAGE_CONTAINER_NAME,
    AZURE_STORAGE_QUEUE_NAME,
    MESSAGES_PER_PAGE,
    MAX_MESSAGES_PER_POLL,
    DELETE_BLOBS,
    AZURE_SEARCH_INDEX_NAME,
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_ADMIN_KEY

)

from python_files.langchain_openai_embeddings import get_vector_store_index

import threading
import atexit
import logging

# Logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(funcName)s - %(lineno)d - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S"
)

def cleanup():
    logging.debug("Starting cleanup of threads.")
    for thread in threading.enumerate():
        logging.debug(f"Checking thread: {thread.name}")
        if thread.is_alive():
            logging.debug(f"Joining thread: {thread.name}")
            thread.join()

atexit.register(cleanup)


async def process_messages_from_blob_storage():
    logging.info("Starting to process blob storage messages.")
    logging.debug(f'Using connection string: {AZURE_STORAGE_URL}')

    queue = QueueClient.from_connection_string(conn_str=AZURE_STORAGE_URL,
                                               queue_name=AZURE_STORAGE_QUEUE_NAME)

    logging.debug("Getting queue properties.")
    properties = queue.get_queue_properties()
    count = properties.approximate_message_count

    logging.debug(f'Queue properties: {properties}')
    logging.debug(f'Approximate message count: {count}')

    if count == 0:
        logging.warning(f'No messages found in the queue (count: {count}). Exiting process.')
        return None
    
    try:
        logging.info(f"Connecting to Azure Queue: {AZURE_STORAGE_QUEUE_NAME}")
        queue.get_queue_properties()
        logging.info("Queue connected successfully.")

        max_messages = max(MESSAGES_PER_PAGE, MAX_MESSAGES_PER_POLL)
        logging.debug(f"Receiving up to {max_messages} messages per page.")
        response = queue.receive_messages(messages_per_page=MESSAGES_PER_PAGE, max_messages=max_messages)

        logging.debug(f'Received response from queue: {response}')

        for batch in response.by_page():
            logging.debug(f"Processing batch: {batch}")
            await process_blobs(client=queue, batch=batch, index_name=AZURE_SEARCH_INDEX_NAME)

    except Exception as e:
        logging.error(f"Error processing messages from blob storage: {e}", exc_info=True)
    finally:
        logging.info("Closing queue connection.")
        queue.close()


async def process_blobs(client: QueueClient, batch: Iterator[QueueMessage], index_name: str):
    logging.debug("Starting blob processing.")
    blob_names, blobs, documents = await split(batch)

    logging.debug(f"Finished splitting documents. Total documents: {len(documents)}")
    logging.debug("Starting document embedding process.")

    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(embed_documents(vector_db=get_vector_store_index(index_name),
                                               documents=docs)) for docs in documents]
            logging.debug(f"Created embedding tasks: {len(tasks)}")
            await asyncio.gather(*tasks)

        logging.info("Text embeddings completed successfully.")
        logging.debug("Deleting processed blob messages from the queue.")

        for blob in blobs:
            logging.debug(f"Deleting message for blob: {blob}")
            client.delete_message(message=blob)

        if DELETE_BLOBS:
            logging.info("Deleting blobs from Azure Blob Storage as per configuration.")
            blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_URL)

            for name in blob_names:
                logging.debug(f'Deleting blob: {name}')
                blob_client = blob_service_client.get_blob_client(container=AZURE_STORAGE_CONTAINER_NAME,
                                                                  blob=name)
                blob_client.delete_blob(delete_snapshots="include")

    except Exception as e:
        logging.error(f"Error during document embedding or deletion: {e}", exc_info=True)


async def split(batch: Iterator[QueueMessage]) -> Tuple[List[str], List[QueueMessage], List[List[Document]]]:
    logging.debug("Splitting messages into individual blobs.")
    messages = []
    blob_names = []

    for message in batch:
        logging.debug(f"Processing message: {message}")
        messages.append(message)

        try:
            blob_path = base64.b64decode(message.content)
            logging.debug(f'Decoded blob path: {blob_path}')
            blob_dict = json.loads(blob_path)
            subject = blob_dict['subject']

            url_parts = urllib.parse.urlparse(subject)
            path_parts = url_parts[2].rpartition('/')
            blob_name: str = path_parts[2]

            logging.debug(f"Extracted blob name: {blob_name}")

            if blob_name.endswith('.pdf'):
                logging.debug(f"Adding PDF to process: {blob_name}")
                blob_names.append(blob_name)
            else:
                logging.warning(f"Blob is not a PDF: {blob_name}")

        except Exception as e:
            logging.error(f"Error processing message: {e}", exc_info=True)

    logging.info(f'Number of messages processed: {len(messages)}')

    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(split_pdf_to_documents(conn_str=AZURE_STORAGE_URL,
                                              container=AZURE_STORAGE_CONTAINER_NAME,
                                              blob_name=name))
                 for name in blob_names]
        logging.debug(f"Created split tasks: {len(tasks)}")
        results = await asyncio.gather(*tasks)

    logging.info(f'Completed processing {len(tasks)} tasks.')
    return blob_names, messages, results

from langchain.text_splitter import RecursiveCharacterTextSplitter

async def split_pdf_to_documents(conn_str: str, container: str, blob_name: str) -> List[Document]:
    logging.debug(f'Starting to split PDF: {blob_name}')
    logging.info(f'Connection string: {conn_str}, Container: {container}, Blob name: {blob_name}')
    
    try:
        logging.debug(f"Connecting to Azure Blob to download: {blob_name}")
        blob_client = BlobClient.from_connection_string(conn_str, container, blob_name)
        pdf_stream = blob_client.download_blob().readall()
        logging.debug(f"Downloaded PDF blob: {blob_name} (size: {len(pdf_stream)} bytes)")

        pdf_file = io.BytesIO(pdf_stream)

        logging.debug(f"Reading PDF: {blob_name}")
        with pdfplumber.open(pdf_file) as pdf:
            num_pages = len(pdf.pages)
            logging.info(f"PDF {blob_name} has {num_pages} pages.")

            documents = []
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            for page_number, page in enumerate(pdf.pages):
                logging.debug(f"Extracting text from page {page_number + 1} of {blob_name}")
                page_text = page.extract_text()

                if not page_text or not page_text.strip():
                    logging.warning(f"Page {page_number + 1} of {blob_name} has empty content.")
                    continue  # Skip this page

                # Split the page text into smaller chunks
                chunks = text_splitter.split_text(page_text)
                for chunk_index, chunk in enumerate(chunks):
                    # Create a Document for each chunk
                    document = Document(
                        page_content=chunk,
                        metadata={
                            'source': blob_name,
                            'page': page_number + 1  # Page numbers start from 1
                        }
                    )
                    logging.debug(f"Created document for page {page_number + 1}, chunk {chunk_index + 1} of {blob_name}")
                    documents.append(document)

        logging.info(f"Successfully split PDF {blob_name} into {len(documents)} documents.")
        return documents

    except Exception as e:
        logging.error(f'Error loading or splitting documents for blob: {blob_name} - {e}', exc_info=True)
        return []

async def embed_documents(vector_db: AzureSearch, documents: List[Document]):
    logging.debug(f"Embedding {len(documents)} documents.")
    try:
        # Extract texts and generate embeddings
        texts = [doc.page_content for doc in documents]
        embeddings = await vector_db.embedding_function.aembed_documents(texts)
        ids = [str(uuid4()) for _ in documents]
        
        # Prepare documents for indexing
        index_documents = []
        for idx, doc in enumerate(documents):
            index_doc = {
                "id": ids[idx],
                "content": texts[idx],
                "content_vector": embeddings[idx],
                "source": doc.metadata.get('source', None),
                "page": int(doc.metadata.get('page', 0)),
                "metadata": json.dumps(doc.metadata)  # Ensure 'metadata' expects a JSON string
            }
            index_documents.append(index_doc)

        # Use the asynchronous SearchClient to upload documents
        async with SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=AZURE_SEARCH_INDEX_NAME,
            credential=AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
        ) as search_client:
            result = await search_client.upload_documents(documents=index_documents)
            logging.info(f"Documents indexed successfully. Upload result: {result}")

    except Exception as e:
        logging.error(f"Error embedding documents: {e}", exc_info=True)