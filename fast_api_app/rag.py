"""Module for the RAG configuration"""
from langchain_community.chat_models import ChatOllama
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from config import Config

class Rag:
    """Given a Retriever, the RAG provides the query related docs as context to the LLM"""
    def __init__(self, retriever=None):
        if retriever is None:
            raise ValueError("Retriever must be set.")

        self.llm = ChatOllama(model=Config.LLM_MODEL)
        self.prompt = hub.pull(Config.RAG_PROMPT_PATH)
        self.retriever = retriever

        self.rag_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def format_docs(self, docs):
        """Format document contents for input to the LLM witt additional metadata"""
        formatted_docs = []
        for doc in docs:
            # Extract CELEX_ID from the document metadata
            celex_id = doc.metadata['_source'].get('CELEX_ID')

            # Prepend CELEX_ID to the document content
            formatted_doc = f"CELEX_ID: {celex_id}\n{doc.page_content}"
            formatted_docs.append(formatted_doc)
        return "\n\n".join(formatted_docs)

    def retrieve_docs_for_chain(self, query: str):
        """Given a query, retrieves the related docs and format them"""
        # Retrieve documents once
        return self.retriever.invoke(query)

    def invoke(self, query: str):
        """Given a query, retrieves related docs and provides them to the chain"""
        # Prepare the context with formatted documents and the query
        docs = self.retrieve_docs_for_chain(query)
        formatted_docs = self.format_docs(docs)
        context = {
            "context": formatted_docs,
            "question": query,
        }

        # Invoke the LLM with the context
        llm_output = self.rag_chain.invoke(context)

        # Return the LLM output along with the CELEX_ID metadata
        return {
            "llm_output": llm_output,
            "related_documents" : docs
        }
