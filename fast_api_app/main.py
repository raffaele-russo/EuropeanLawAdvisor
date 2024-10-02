"""
    European Law Advisor
"""
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse


# Create the FastAPI app
app = FastAPI()

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

@app.post("/", response_class=HTMLResponse)
async def handle_search(request: Request, query: str = Form(...)):
    """
    Handle search query from the form and return results via 'index.html' template.
    """
    # Render the template with search results (empty for now)
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request,
            "query": query,
            "results": [],
            "from_": 0,
            "total": 0
        }
    )

@app.get("/document/{id_document}", response_class=JSONResponse)
async def get_document(id_document: int):
    """
    Get document by ID.
    """
    return {"message": f"Document with ID {id_document} not found"}
