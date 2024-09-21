import logging
from langchain_openai import ChatOpenAI
from python_files.constants import (
    OPENAI_API_KEY,
    AZURE_OPENAI_API_ENDPOINT,
    AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME,
    AZURE_OPENAI_API_EMBEDDINGS_VERSION,
    AZURE_OPENAI_EMBEDDINGS_MODEL_NAME,
    AZURE_OPENAI_API_KEY
)
from langchain_openai import AzureOpenAIEmbeddings

# Logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(funcName)s - %(lineno)d - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S"
)

# Function to create an OpenAI chat client using LangChain
def create_openai_client(temperature: float = 0.0):
    """
    Create an OpenAI chat client for generating chat completions using OpenAI API.
    
    Args:
        temperature (float): Sampling temperature for randomness in completions.

    Returns:
        ChatOpenAI: LangChain's ChatOpenAI client.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini-2024-07-18", 
        api_key=OPENAI_API_KEY,  # Use openai_api_key
        temperature=temperature
    )
    return llm

def create_azure_embeddings() -> AzureOpenAIEmbeddings:
    """
    Create an Azure OpenAI Embeddings object using Azure OpenAI API.
    
    Returns:
        AzureOpenAIEmbeddings: LangChain's AzureOpenAIEmbeddings object.
    """
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_OPENAI_API_ENDPOINT,
        azure_deployment=AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME,
        openai_api_version=AZURE_OPENAI_API_EMBEDDINGS_VERSION,
        model=AZURE_OPENAI_EMBEDDINGS_MODEL_NAME,
        api_key=AZURE_OPENAI_API_KEY
    )
    return embeddings

# Function to create chat completions using OpenAI API
def create_chat_completion(messages: list):
    """
    Create a chat completion using OpenAI's API.

    Args:
        messages (list): A list of conversation messages for generating a response.

    Returns:
        str: The assistant's response.
    """
    try:
        client = create_openai_client(temperature=0.7)
        response = client(messages=messages)  # Updated API call for LangChain's ChatOpenAI
        return response['choices'][0]['message']['content']
    except Exception as e:
        logging.error(f"Error in create_chat_completion: {e}", exc_info=True)
        return "Error generating response."