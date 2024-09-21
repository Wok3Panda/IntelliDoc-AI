import os
from dotenv import load_dotenv

def strip_quotes(value):
    if value:
        return value.strip().strip('"')
    return value


# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=os.getenv("ENV_FILE"))
print(f'OS ENV Configuration File - {dotenv_path}')

GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", 8888))

# Azure OpenAI
AZURE_OPENAI_API_TYPE = "azure"
AZURE_OPENAI_MODEL_NAME = os.getenv("AZURE_OPENAI_MODEL_NAME", None)
AZURE_OPENAI_API_ENDPOINT = os.getenv("AZURE_OPENAI_API_ENDPOINT", "gpt-4")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", None)
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", None)
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME = os.getenv(
    "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME", None
)
AZURE_OPENAI_EMBEDDINGS_MODEL_NAME = os.getenv(
    "AZURE_OPENAI_EMBEDDINGS_MODEL_NAME", None
)
AZURE_OPENAI_API_EMBEDDINGS_VERSION = os.getenv("AZURE_OPENAI_API_EMBEDDINGS_VERSION", "2024-02-01")

# Open AI API Info
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_COMPLETION_MODEL = os.getenv("OPENAI_COMPLETION_MODEL", "gpt-4o-mini-2024-07-18")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

# Azure Search Info
AZURE_SEARCH_ENDPOINT =os.getenv("AZURE_SEARCH_ENDPOINT", None)
AZURE_SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY", None)
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", None)

# LLM Type
LLM_TYPE = os.getenv("LLM_TYPE", 'Azure')

# AZURE STORAGE INFO
AZURE_STORAGE_URL = strip_quotes(os.getenv("AZURE_STORAGE_URL", None))
AZURE_STORAGE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME", None)
AZURE_STORAGE_QUEUE_NAME = os.getenv("AZURE_STORAGE_QUEUE_NAME", None)

MESSAGES_PER_PAGE = int(os.getenv("MESSAGES_PER_PAGE", 10))
MAX_MESSAGES_PER_POLL = int(os.getenv("MAX_MESSAGES_PER_POLL", 50))
DELETE_BLOBS = os.getenv("DELETE_BLOBS", False)
CRON_EXPRESSION = os.getenv("CRON_EXPRESSION", "0 0/2 * * *")
print(f'CRON_EXPRESSION: {CRON_EXPRESSION}')
print(f'MAX_MESSAGES_PER_POLL: {MAX_MESSAGES_PER_POLL}')
print(f'MESSAGES_PER_PAGE: {MESSAGES_PER_PAGE}')

APP_TITLE = os.getenv("APP_TITLE", None)

APP_VERSION = os.getenv("APP_VERSION", None)

MAX_CHAT_HISTORY_LENGTH = int(os.getenv("MAX_CHAT_HISTORY_LENGTH", 5))