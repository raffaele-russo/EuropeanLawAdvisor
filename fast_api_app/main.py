"""
    European Law Advisor
"""
import logging
import joblib
from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from search import Search
from rag import Rag
from config import Config
from app_logging import setup_logging

# Setup logging
logger = logging.getLogger(__name__)
setup_logging(logger)

# Create the FastAPI app
app = FastAPI()
logger.info("Starting European Law Advisor app")

try:
    # Elastic Search Client
    es = Search()
    es.vectorizer = joblib.load(Config.VECTORIZER_MODEL_PATH)
    logger.info("ElasticSearch client initialized successfully")
except Exception as e:
    logger.error("Error initializing ElasticSearch client: %s", e)
    raise HTTPException(
        status_code=500,
        detail="Internal server error during Elasticsearch init"
        ) from e

try:
    rag = Rag(es.retriever)
    logger.info("RAG initialized successfully")
except Exception as e:
    logger.error("Error initializing RAG: %s",e)
    raise HTTPException(
        status_code=500,
        detail="Internal server error during RAG init"
        ) from e

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

@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...), query_type: str = Form(...),
                 from_ : int = Form(0)):
    """
    Handle search query from the form and return results via 'index.html' template.
    """
    logger.info("Received search request: query=%s, type=%s", query, query_type)
    results = []

    try:
        if query_type == "rag":
            answer = rag.invoke(query)
            results = [
                {**doc.metadata, "page_content": doc.page_content}
                for doc in answer["related_documents"]
            ]
            logger.info("RAG search returned %s results", len(results))
            return templates.TemplateResponse(
                "index.html", 
                {
                    "request": request, 
                    "results":results, 
                    "answer": answer, 
                    "from_": 0, 
                    "total": len(results)
                })

        if query_type == "multi_match":
            logger.info("Performing multi-match search")
            results = es.multi_match_search(query = query,
                                            query_fields = es.get_text_index_fields(),
                                            from_=from_)
        elif query_type == "knn":
            logger.info("Performing knn search")
            results = es.knn_search(query, from_=from_)
        elif query_type == "hybrid":
            logger.info("Performing hybrid search")
            results = es.hybrid_search(query = query,
                                    query_fields = es.get_text_index_fields(),
                                    from_=from_)
        elif query_type == "tf-idf":
            logger.info("Performing semantic search (TF-IDF)")
            results = es.semantic_search(query, from_=from_)

        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request,
                "query": query,
                "results": results['hits']['hits'],
                "from_": from_,
                "query_type" : query_type,
                "total": results['hits']['total']['value']
            }
        )
    except Exception as e:
        logger.error("Error during search operation: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Internal server error during search operation"
            ) from e

@app.get("/document/{id_document}", response_class=HTMLResponse)
async def get_document(request: Request, id_document: int):
    """
    Get document by ID.
    """
    logger.info("Fetching document with ID: %s", id_document)
    try:
        document = es.retrieve_document(id_document)
        celex_id = document['_source']['CELEX_ID']
        text = document['_source'][Config.QUERY_TEXT_FIELD]
        return templates.TemplateResponse(
            "document.html", 
            {
                "request": request,
                "celex_id": celex_id,
                "text": text
            }
        )
    except Exception as e:
        logger.error("Error retrieving document with ID %s: %s", id_document, e)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error retrieving document {id_document}"
            ) from e
