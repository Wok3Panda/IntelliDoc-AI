import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.triggers.cron import CronTrigger
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage

from python_files.langchain_openai_embeddings import get_retriever_with_chat_history
from python_files.document_ingestion_scheduler import process_messages_from_blob_storage
from python_files.doc_ui import demo
from python_files.constants import (
    CRON_EXPRESSION,
    MAX_CHAT_HISTORY_LENGTH,
    AZURE_SEARCH_INDEX_NAME,
    APP_TITLE,
    APP_VERSION,
    GRADIO_SERVER_PORT
)

from python_files.version_router import router

os.environ['TZ'] = 'America/New_York'

# Logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(funcName)s - %(lineno)d - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S"
)

logging.debug("Setting up AsyncIOScheduler")
scheduler = AsyncIOScheduler()


# Add this function to lifespan, only if you want the document ingestion to begin as soon as the code is executed
def start_scheduler():
    logging.debug("Adding job to scheduler")
    scheduler.add_job(process_messages_from_blob_storage, CronTrigger.from_crontab(CRON_EXPRESSION))
    logging.debug("Starting scheduler")
    scheduler.start()
    logging.info("Scheduler has been started and job scheduled.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Starting application...")

    # Start Gradio app as a background task
    gradio_task = asyncio.create_task(run_gradio())
    
    try:
        yield
    finally:
        logging.info("Shutting down application...")
        # Cancel Gradio task
        gradio_task.cancel()
        try:
            await gradio_task
        except asyncio.CancelledError:
            logging.info("Gradio app has been shut down.")

description = """
  A Generative AI application for (i) Querying scientific documents
"""

logging.debug("Creating FastAPI app")
app = FastAPI(lifespan=lifespan, title=APP_TITLE, version=APP_VERSION, description=description)

logging.debug("Including router")
app.include_router(router)

logging.debug("Adding CORS middleware")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"],
                   allow_headers=["*"])

class Prompt(BaseModel):
    prompt: str = Field(description="AI Prompt")

    chat_history: list[dict] = Field(description="Array of JSON containing chat history",
                                     examples=["""
                                          'chat_history': [
                                               {"Do children get serious COVID-19": "No"},
                                               {"Did countries close schools during COVID-19": "Yes"},
                                          ]
                                     """])

    def get_chat_history(self):
        rv = []
        for history in self.chat_history:
            for k, v in history.items():
                rv.append(HumanMessage(content=k))
                rv.append(AIMessage(content=v))

        # Restrict the chat history to max allowed
        if len(rv) > MAX_CHAT_HISTORY_LENGTH * 2:
            rv = rv[-MAX_CHAT_HISTORY_LENGTH * 2:]

        return rv

@app.get("/", description='Health check endpoint')
async def index():
    logging.debug("Health check endpoint called")
    return "Welcome to Valeos Transplant Document Summarizer"

@app.post("/ask")
async def ask(prompt: Prompt):
    logging.debug(f"Received ask request with prompt: {prompt.prompt}")
    retriever = get_retriever_with_chat_history(index_name=AZURE_SEARCH_INDEX_NAME)
    llm_response = await retriever.invoke({"input": prompt.prompt, "chat_history": prompt.get_chat_history()})
    logging.debug(f"LLM response: {llm_response}")
    return llm_response['answer']

async def run_gradio():
    logging.debug("Starting Gradio app")
    demo.launch(
        server_name="0.0.0.0", 
        server_port=GRADIO_SERVER_PORT, 
        share=False, 
        inbrowser=False, 
        show_error=True, 
        prevent_thread_lock=True
    )

# command to run uvicorn - uvicorn server:app --host 0.0.0.0 --port 8000 --reload
# For Local development visit http://localhost:8888/ after execting the above command
if __name__ == "__main__":
    import uvicorn

    logging.info("Starting the server with Uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000)