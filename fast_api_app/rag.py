"""Module for the Retrieval-Augmented Generation (RAG) configuration."""

import logging
from typing import List, Dict, Any
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAI
import tiktoken
from helper.config import Config
from helper.app_logging import setup_logging
from helper.exception_handler import exception_handler

# Setup logging
logger = logging.getLogger(__name__)
setup_logging(logger)

class Rag:
    """Retrieval-Augmented Generation (RAG) class that integrates a retriever and an LLM.
    
    This class retrieves relevant documents based on a query and provides them
    as context for the Language Model (LLM) to generate responses.
    """

    def __init__(self, es: Any) -> None:
        """
        Initializes the RAG system with a given Elasticsearch instance.

        Args:
            es: An Elasticsearch instance used for document retrieval.
        """
        self.llm = self.initialize_llm()
        self.prompt = self.initialize_prompt()
        self.retriever = self.initialize_retriever(es)
        self.chunks_retriever = self.initialize_chunks_retriever(es)
        self.tokenizer = self.initialize_tokenizer()
        self.rag_chain = self.initialize_rag_chain()

    @exception_handler("No valid LLM configuration provided")
    def initialize_llm(self) -> Any:
        """Initialize the Language Model (LLM) based on the configuration.

        Returns:
            The initialized LLM instance.

        Raises:
            RagException: If no valid LLM configuration is provided or initialization fails.
        """
        if Config.OPENAI_API_KEY:
            llm = OpenAI(model=Config.OPENAI_MODEL_NAME)
            logger.info("Using OpenAI model: %s", Config.OPENAI_MODEL_NAME)
            return llm

        llm = ChatOllama(model=Config.LLM_MODEL)
        logger.info("Using ChatOllama model: %s", Config.LLM_MODEL)
        return llm

    @exception_handler("Error in prompt initialization")
    def initialize_prompt(self) -> PromptTemplate:
        """Initialize the prompt from a custom or default path.

        Returns:
            PromptTemplate: The initialized prompt template.

        Raises:
            RagException: If the prompt initialization fails.
        """
        if Config.RAG_CUSTOM_PROMPT_PATH:
            logger.info("Relying on custom prompt: %s", Config.RAG_CUSTOM_PROMPT_PATH)
            return PromptTemplate.from_file(Config.RAG_CUSTOM_PROMPT_PATH)
        logger.info("Relying on default prompt %s", Config.RAG_PROMPT_PATH)
        return hub.pull(Config.RAG_PROMPT_PATH)

    @exception_handler("Failed to initialize retriever")
    def initialize_retriever(self, es: Any) -> Any:
        """Initialize the document retriever.

        Args:
            es: An Elasticsearch instance used for document retrieval.

        Returns:
            The initialized document retriever.

        Raises:
            RagException: If the retriever initialization fails.
        """
        return es.retriever

    @exception_handler("Failed to initialize chunks retriever")
    def initialize_chunks_retriever(self, es: Any) -> Any:
        """Initialize the chunks retriever.

        Args:
            es: An Elasticsearch instance used for chunk retrieval.

        Returns:
            The initialized chunks retriever.

        Raises:
            RagException: If the chunks retriever initialization fails.
        """
        return es.vector_store.as_retriever(
                search_kwargs={'k': Config.K_CHUNKS_TO_RETRIEVE})

    @exception_handler("Failed to initialize tokenizer")
    def initialize_tokenizer(self) -> Any:
        """Initialize the tokenizer for the specified model.

        Returns:
            The initialized tokenizer.

        Raises:
            RagException: If the tokenizer initialization fails.
        """
        return tiktoken.encoding_for_model(Config.OPENAI_MODEL_NAME)

    @exception_handler("Failed to initialize RAG chain")
    def initialize_rag_chain(self) -> Any:
        """Initialize the RAG chain for processing.

        Returns:
            The initialized RAG chain.

        Raises:
            RagException: If the RAG chain initialization fails.
        """
        return (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
        )

    @exception_handler("An issue occurred when computing token number")
    def count_tokens(self, text: str) -> int:
        """Counts the number of tokens in the provided text.

        Args:
            text: The text to count tokens for.

        Returns:
            int: The number of tokens in the text.

        Raises:
            RagException: If an error occurs while counting tokens.
        """
        return len(self.tokenizer.encode(text))

    @exception_handler("An error occured when ensuring token limit")
    def ensure_token_limit(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Trims the context chunks to ensure the total token count stays within the limit.

        Args:
            context: A dictionary containing the context and query.

        Returns:
            The updated context with trimmed chunks.

        Raises:
            RagException: If an error occurs while ensuring the token limit.
        """
        query_tokens = self.count_tokens(context["question"])

        # Convert each chunk to its token length
        chunk_tokens = [(chunk, self.count_tokens(chunk)) for chunk in context["context"]]
        total_tokens = query_tokens + sum(tokens for _, tokens in chunk_tokens)

        # If the total tokens exceed the max limit, remove chunks from the context
        while total_tokens > Config.MAX_TOKENS:
            logger.info("Context too long, trimming...")
            # Remove the first chunk (or adjust as needed) and recalculate total tokens
            removed_chunk, removed_tokens = chunk_tokens.pop(0)
            logger.debug("Remove chunk: %s", removed_chunk)
            total_tokens -= removed_tokens

        # Rebuild the context with the trimmed chunks
        trimmed_chunks = [chunk for chunk, _ in chunk_tokens]
        context["context"] = trimmed_chunks

        return context

    @exception_handler("An issue occurred when formatting the docs")
    def format_docs(self, docs: List[Any]) -> str:
        """Format document contents for input to the LLM with additional metadata.

        Args:
            docs: A list of documents to format.

        Returns:
            str: A string containing formatted documents.

        Raises:
            RagException: If an error occurs while formatting documents.
        """
        formatted_docs = []
        for doc in docs:
            # Extract CELEX_ID from the document metadata
            celex_id = doc.metadata['_source'].get('CELEX_ID')

            # Prepend CELEX_ID to the document content
            formatted_doc = f"CELEX_ID: {celex_id}\n{doc.page_content}"
            formatted_docs.append(formatted_doc)

        return "\n\n".join(formatted_docs)

    @exception_handler("An issue occurred when formatting the chunks")
    def format_chunks(self, chunks: List[Any]) -> List[str]:
        """Format chunks for input to the LLM with additional metadata.

        Args:
            chunks: A list of chunks to format.

        Returns:
            List[str]: A list of formatted chunks.

        Raises:
            RagException: If an error occurs while formatting chunks.
        """
        formatted_chunks = []
        for chunk in chunks:
            # Extract CELEX_ID from the document metadata
            celex_id = chunk.metadata.get('CELEX_ID')

            # Prepend CELEX_ID to the document content
            formatted_chunk = f"CELEX_ID: {celex_id}\n{chunk.page_content}"
            formatted_chunks.append(formatted_chunk)
        return formatted_chunks

    @exception_handler("An issue occurred when retrieving docs for query")
    def retrieve_related_docs(self, query: str) -> List[Any]:
        """Given a query, retrieves the related documents.

        Args:
            query: The query string to use for document retrieval.

        Returns:
            List[Any]: A list of related documents.

        Raises:
            RagException: If an error occurs during document retrieval.
        """
        return self.retriever.invoke(query)

    @exception_handler("An issue occured when retrieving chunks given the query")
    def retrieve_related_chunks_for_context(self, query: str) -> List[Any]:
        """Given a query, retrieves the related chunks for the context.

        Args:
            query: The query string to use for context retrieval.

        Returns:
            List[Any]: A list of related chunks.

        Raises:
            RagException: If an error occurs during chunk retrieval.
        """
        return self.chunks_retriever.invoke(query)

    @exception_handler("An issue occurred during the invocation")
    def invoke(self, query: str) -> Dict[str, Any]:
        """Given a query, retrieves related documents and 
           provides them to the chain for LLM processing.

        Args:
            query: The query string to process.

        Returns:
            Dict[str, Any]: A dictionary containing the LLM output and related documents.

        Raises:
            RagException: If an error occurs during invocation.
        """
        related_docs = self.retrieve_related_docs(query)
        chunked_docs = self.retrieve_related_chunks_for_context(query)
        formatted_chunks = self.format_chunks(chunked_docs)
        context = {
            "context": formatted_chunks,
            "question": query,
        }

        logger.debug("LLM context: %s", context)

        # Ensure the total token count is within the maximum allowed
        context = self.ensure_token_limit(context)

        logger.debug("LLM trimmed context: %s", context)

        # Invoke the LLM with the context
        llm_output = self.rag_chain.invoke(context, max_tokens = -1)
        logger.info("LLM has provided an answer")
        logger.debug("LLM output: %s", llm_output)

        # Return the LLM output along with the related documents
        return {
            "llm_output": llm_output,
            "related_documents": related_docs
        }
