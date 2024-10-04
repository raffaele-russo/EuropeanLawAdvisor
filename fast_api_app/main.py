"""
    European Law Advisor
"""
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from search import Search


# Create the FastAPI app
app = FastAPI()

# Create Elastic Search client
es = Search()

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Index endpoint that renders the 'index.html' template.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...), query_type: str = Form(...),
                 from_ : int = Form(0)):
    """
    Handle search query from the form and return results via 'index.html' template.
    """
    results = []
    if query_type == "multi_match":
        query_fields = es.get_text_index_fields()
        results = es.multi_match_search(query = query, query_fields = query_fields, from_=from_)
    elif query_type == "knn":
        results = es.knn_search(query)

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

@app.get("/document/{id_document}", response_class=HTMLResponse)
async def get_document(request: Request, id_document: int):
    """
    Get document by ID.
    """
    document = es.retrieve_document(id_document)
    celex_id = document['_source']['CELEX_ID']
    text = document['_source']['Text']
    return templates.TemplateResponse(
        "document.html", 
        {
            "request": request,
            "celex_id": celex_id,
            "text": text
        }
    )
