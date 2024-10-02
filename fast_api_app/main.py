"""
A simple FastAPI application that returns 'Hello World'.
"""

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    """
    Root endpoint returning a 'Hello World' message.
    """
    return {"message": "Hello World"}
