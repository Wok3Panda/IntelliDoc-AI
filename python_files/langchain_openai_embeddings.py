import logging
from azure.storage.blob import BlobServiceClient
from azure.search.documents.indexes.models import SimpleField, SearchableField, SearchField, SearchFieldDataType

from langchain_community.vectorstores.azuresearch import AzureSearch

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from typing import Any, List

from python_files.langchain_llm_utils import (
    create_openai_client,
    create_azure_embeddings,
)

from python_files.constants import (
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_ADMIN_KEY,
    AZURE_SEARCH_INDEX_NAME,
    AZURE_STORAGE_URL,
    AZURE_STORAGE_CONTAINER_NAME
)

from azure.search.documents.indexes.models import SearchFieldDataType

def get_vector_store_index(index_name: str) -> AzureSearch:
    logging.debug(f"Creating AzureSearch with index_name: {index_name}")
    
    # Create the Azure OpenAI embeddings model
    embeddings_model = create_azure_embeddings()
    logging.debug("Created Azure OpenAI Embeddings model")

    # Return the configured AzureSearch object
    return AzureSearch(
        azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
        azure_search_key=AZURE_SEARCH_ADMIN_KEY,
        index_name=index_name,
        embedding_function=embeddings_model
    )

def get_retriever(index_name: str):

    logging.debug(f"Getting retriever for index_name: {index_name}")
    
    llm = create_openai_client(temperature=0.0)
    
    retriever = get_vector_store_index(index_name).as_retriever()
    
    # Updated template with {context}
    template = """Answer the question based only on the provided context: {context}

[Response]

Reference:

If there is no context provided, then just state that there are no references available to answer the question. No further processing is required.
"""

    prompt = ChatPromptTemplate.from_template(template)

    output_parser = StrOutputParser()

    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )
    
    chain = setup_and_retrieval | prompt | llm | output_parser

    return chain


def get_retriever_with_chat_history(index_name: str):
    logging.debug(f"Getting retriever with chat history for index_name: {index_name}")

    contextualize_q_system_prompt = """
       Given a chat history and the latest user question \
       which might reference context in the chat history, formulate a standalone question \
       which can be understood without the chat history. Do NOT answer the question, \
       just reformulate it if needed and otherwise return it as is.
    """

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    llm = create_openai_client(temperature=0.0)

    db_retriever = get_vector_store_index(index_name).as_retriever()

    history_aware_retriever = create_history_aware_retriever(
        llm, db_retriever, contextualize_q_prompt
    )

    # Updated template with {context}
    template = """Answer the question based only on the provided context: {context}

[Response]

Reference:

If there is no context provided, then just state that there are no references available to answer the question. No further processing is required.
"""

    qa_prompt = ChatPromptTemplate.from_template(template)

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain



async def get_number_of_documents() -> int:
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_URL)
    container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER_NAME)
    
    blob_list = container_client.list_blobs()
    return sum(1 for _ in blob_list)  # Count the number of blobs

def get_azure_ai_fields() -> list[Any]:
    embedding_model = create_azure_embeddings()
    sample_embedding = embedding_model.embed_query("Sample text")
    vector_search_dim = len(sample_embedding)
    return [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            searchable=True,
        ),
        SearchableField(
            name="metadata",
            type=SearchFieldDataType.String,
            searchable=True,
            retrievable=True
        ),
        SearchField(
            name="page",
            type=SearchFieldDataType.Int32,
            searchable=False,
            filterable=True,
            retrievable=True
        ),
        SearchableField(
            name="source",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=True,
            retrievable=True
        ),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=vector_search_dim,
            vector_search_configuration="default",
        )
    ]

if __name__ == "__main__":

    retriever = get_retriever(index_name=AZURE_SEARCH_INDEX_NAME)