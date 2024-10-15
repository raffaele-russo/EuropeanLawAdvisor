from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser   
from config import Config

def format_docs(docs):
    """Format document contents for input to the LLM without losing metadata, including CELEX_ID."""
    formatted_docs = []
    for doc in docs:
        # Extract CELEX_ID from the document metadata
        celex_id = doc.metadata['_source'].get('CELEX_ID')

        # Prepend CELEX_ID to the document content
        formatted_doc = f"CELEX_ID: {celex_id}\n{doc.page_content}"  
        formatted_docs.append(formatted_doc)

    return "\n\n".join(formatted_docs)  # Join documents with double newlines

class Rag:
    def __init__(self, retriever=None):
        if retriever is None:
            raise ValueError("Retriever must be set.")

        self.llm = ChatOllama(model=Config.LLM_MODEL)
        self.prompt = hub.pull(Config.RAG_PROMPT_PATH)
        self.retriever = retriever

        self.rag_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}  # Placeholder for dynamic context
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def invoke(self, query: str):
        # Retrieve documents once
        docs = self.retriever.invoke(query)

        # Save metadata for each document
        celex_metadata = []
        documents_ids = []

        for doc in docs:
            celex_metadata.append(doc.metadata['_source'].get('CELEX_ID', 'Unknown CELEX_ID'))
            documents_ids.append(doc.metadata['_id'])

        # Format the documents for the LLM
        formatted_docs = format_docs(docs)

        # Prepare the context with formatted documents and the query
        context = {
            "context": formatted_docs,
            "question": query,
        }

        # Invoke the LLM with the context
        llm_output = self.rag_chain.invoke(context)

        # Return the LLM output along with the CELEX_ID metadata
        return {
            "llm_output": llm_output,
            "celex_metadata": celex_metadata,
            "related_documents_ids": documents_ids,
            "related_documents" : docs
        }

