"""
    European Law Advisor
"""
import logging
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from search import Search
from rag import Rag
from helper.config import Config
from helper.exception_handler import exception_handler
from helper.app_logging import setup_logging

# Suppressing pylint warning for broad-except as exceptions are handled gracefully
# The goal is to provide a better user experience without exposing technical errors

# Setup logging
logger = logging.getLogger(__name__)
setup_logging(logger)

# Create the FastAPI app
app = FastAPI()
logger.info("Starting European Law Advisor app")

# Create Elasticsearch client and RAG
es = Search()
es.load_weights()
rag = Rag(es)

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Index endpoint that renders the 'index.html' template.
    """
    logger.info("Rendering index page")
    return templates.TemplateResponse("index.html", {"request": request})

@exception_handler("Error occurred while rendering the response")
async def render_response(request: Request, **kwargs : dict) -> HTMLResponse:
    """
    Helper function to render template responses.
    """
    return templates.TemplateResponse("index.html", {"request": request, **kwargs})

@exception_handler("Error occurred while performing RAG query")
async def perform_rag_query(query: str) -> dict:
    """
    Provides query to RAG and returns a formatted response
    """
    logger.info("Sending request to llm")
    answer = rag.invoke(query)
    results = [{**doc.metadata, "page_content": doc.page_content}
    for doc in answer["related_documents"]]
    logger.info("RAG search returned %s results", len(results))
    return results, answer

@exception_handler("Error occurred while performing search query")
async def perform_search_query(query: str, query_type: str, from_: int) -> dict:
    """
    Performs the specified search query
    """
    query_types = Config.QUERY_TYPES
    if query_type == query_types['MULTI_MATCH']:
        logger.info("Performing multi-match search")
        return es.multi_match_search(query=query,
        query_fields=es.get_text_index_fields(), from_=from_)
    if query_type == query_types['KNN']:
        logger.info("Performing knn search")
        return es.knn_search(query, from_=from_)
    if query_type == query_types['HYBRID']:
        logger.info("Performing hybrid search")
        return es.hybrid_search(query=query, query_fields=es.get_text_index_fields(), from_=from_)
    if query_type == query_types['TF_IDF']:
        logger.info("Performing semantic search (TF-IDF)")
        return es.semantic_search(query, from_=from_)

    logger.error("Unknown query type: %s", query_type)
    raise ValueError("Invalid query type")

@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...),
query_type: str = Form(...), from_: int = Form(0)):
    """
    Handle search query from the form and return results via 'index.html' template.
    """
    logger.info("Received search request: query=%s, type=%s", query, query_type)

    try:
        if query_type == Config.QUERY_TYPES['RAG']:
            results, answer = await perform_rag_query(query)
            return await render_response(
                request=request,
                results=results,
                query=query,
                answer=answer,
                from_=0,
                total=len(results)
            )

        results = await perform_search_query(query, query_type, from_)

        return await render_response(
            request=request,
            query=query,
            results=results['hits']['hits'],
            from_=from_,
            query_type=query_type,
            total=results['hits']['total']['value']
        )
    except Exception as e: # pylint: disable=broad-except
        logger.error("An error occurred while performing the query: %s", e)
        return await render_response(
            request=request,
            results=None,
            answer=None,
            error_message="An error occurred while processing the request. Please try again."
        )
