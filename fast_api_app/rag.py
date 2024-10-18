"""Module for the Retrieval-Augmented Generation (RAG) configuration."""

import logging
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from config import Config
from langchain_openai import OpenAI
import tiktoken
from app_logging import setup_logging
from typing import List, Dict, Any

# Setup logging
logger = logging.getLogger(__name__)
setup_logging(logger)

class RagException(Exception):
    """Base class for exceptions in the RAG module."""
    def __init__(self, message: str) -> None:
        super().__init__(message)

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

    def initialize_llm(self) -> Any:
        """Initialize the Language Model (LLM) based on the configuration.

        Returns:
            The initialized LLM instance.

        Raises:
            RagException: If no valid LLM configuration is provided or initialization fails.
        """
        try:
            if Config.OPENAI_API_KEY:
                llm = OpenAI(model=Config.OPENAI_MODEL_NAME)
                logger.info("Using OpenAI model: %s", Config.OPENAI_MODEL_NAME)
                return llm
            elif Config.LLM_MODEL:
                llm = ChatOllama(model=Config.LLM_MODEL)
                logger.info("Using ChatOllama model: %s", Config.LLM_MODEL)
                return llm
            else:
                raise RagException("No valid LLM configuration provided")
        except Exception as e:
            logger.error("Failed to initialize LLM: %s", str(e))
            raise RagException(f"Failed to initialize LLM: {str(e)}")

    def initialize_prompt(self) -> PromptTemplate:
        """Initialize the prompt from a custom or default path.

        Returns:
            PromptTemplate: The initialized prompt template.

        Raises:
            RagException: If the prompt initialization fails.
        """
        try:
            if Config.RAG_CUSTOM_PROMPT_PATH:
                logger.info("Relying on custom prompt: %s", Config.RAG_CUSTOM_PROMPT_PATH)
                return PromptTemplate.from_file(Config.RAG_CUSTOM_PROMPT_PATH)
            else:
                logger.info("Relying on default prompt %s", Config.RAG_PROMPT_PATH)
                return hub.pull(Config.RAG_PROMPT_PATH)
        except Exception as e:
            logger.error("Failed to initialize prompt: %s", str(e))
            raise RagException(f"Failed to initialize prompt: {str(e)}")

    def initialize_retriever(self, es: Any) -> Any:
        """Initialize the document retriever.

        Args:
            es: An Elasticsearch instance used for document retrieval.

        Returns:
            The initialized document retriever.

        Raises:
            RagException: If the retriever initialization fails.
        """
        try:
            return es.retriever 
        except Exception as e:
            logger.error("Failed to initialize retriever: %s", str(e))
            raise RagException(f"Failed to initialize retriever: {str(e)}")

    def initialize_chunks_retriever(self, es: Any) -> Any:
        """Initialize the chunks retriever.

        Args:
            es: An Elasticsearch instance used for chunk retrieval.

        Returns:
            The initialized chunks retriever.

        Raises:
            RagException: If the chunks retriever initialization fails.
        """
        try:
            return es.vector_store.as_retriever(search_kwargs={'k': Config.K_CHUNKS_TO_RETRIEVE})
        except Exception as e:
            logger.error("Failed to initialize chunks retriever: %s", str(e))
            raise RagException(f"Failed to initialize chunks retriever: {str(e)}")
    
    def initialize_tokenizer(self) -> Any:
        """Initialize the tokenizer for the specified model.

        Returns:
            The initialized tokenizer.

        Raises:
            RagException: If the tokenizer initialization fails.
        """
        try:
            return tiktoken.encoding_for_model(Config.OPENAI_MODEL_NAME)
        except Exception as e:
            logger.error("Failed to initialize tokenizer: %s", str(e))
            raise RagException(f"Failed to initialize tokenizer: {str(e)}")

    def initialize_rag_chain(self) -> Any:
        """Initialize the RAG chain for processing.

        Returns:
            The initialized RAG chain.

        Raises:
            RagException: If the RAG chain initialization fails.
        """
        try:
            return (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
        except Exception as e:
            logger.error("Failed to initialize RAG chain: %s", str(e))
            raise RagException(f"Failed to initialize RAG chain: {str(e)}")

    def count_tokens(self, text: str) -> int:
        """Counts the number of tokens in the provided text.

        Args:
            text: The text to count tokens for.

        Returns:
            int: The number of tokens in the text.

        Raises:
            RagException: If an error occurs while counting tokens.
        """
        try:
            token_number = len(self.tokenizer.encode(text))
        except Exception as e:
            logger.error("An issue occurred when computing the token number: %s", str(e))
            raise RagException(f"An issue occurred when computing the token number: {str(e)}")
        return token_number
    
    def ensure_token_limit(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Trims the context chunks to ensure the total token count stays within the limit.

        Args:
            context: A dictionary containing the context and query.

        Returns:
            The updated context with trimmed chunks.

        Raises:
            RagException: If an error occurs while ensuring the token limit.
        """
        try:
            query_tokens = self.count_tokens(context["question"])
            
            # Convert each chunk to its token length
            chunk_tokens = [(chunk, self.count_tokens(chunk)) for chunk in context["context"]]

            total_tokens = query_tokens + sum(tokens for _, tokens in chunk_tokens)

            # If the total tokens exceed the max limit, remove chunks from the context
            while total_tokens > Config.MAX_TOKENS:
                logger.info("Context too long, trimming...")
                # Remove the first chunk (or adjust as needed) and recalculate total tokens
                removed_chunk, removed_tokens = chunk_tokens.pop(0)
                total_tokens -= removed_tokens

            # Rebuild the context with the trimmed chunks
            trimmed_chunks = [chunk for chunk, _ in chunk_tokens]
            context["context"] = trimmed_chunks

        except Exception as e:
            logger.error("An issue occurred when ensuring the token limit: %s", str(e))
            raise RagException(f"An issue occurred when ensuring the token limit: {str(e)}")
        return context

    def format_docs(self, docs: List[Any]) -> str:
        """Format document contents for input to the LLM with additional metadata.

        Args:
            docs: A list of documents to format.

        Returns:
            str: A string containing formatted documents.

        Raises:
            RagException: If an error occurs while formatting documents.
        """
        try:
            formatted_docs = []
            for doc in docs:
                # Extract CELEX_ID from the document metadata
                celex_id = doc.metadata['_source'].get('CELEX_ID')

                # Prepend CELEX_ID to the document content
                formatted_doc = f"CELEX_ID: {celex_id}\n{doc.page_content}"
                formatted_docs.append(formatted_doc)
        except Exception as e:
            logger.error("An issue occurred when formatting the docs: %s", str(e))
            raise RagException(f"An issue occurred when formatting the docs: {str(e)}")
        return "\n\n".join(formatted_docs)
    
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
        try:
            for chunk in chunks:
                # Extract CELEX_ID from the document metadata
                celex_id = chunk.metadata.get('CELEX_ID')

                # Prepend CELEX_ID to the document content
                formatted_chunk = f"CELEX_ID: {celex_id}\n{chunk.page_content}"
                formatted_chunks.append(formatted_chunk)
        except Exception as e:
            logger.error("An issue occurred when formatting the chunks: %s", str(e))
            raise RagException(f"An issue occurred when formatting the chunks: {str(e)}")
        return formatted_chunks

    def retrieve_related_docs(self, query: str) -> List[Any]:
        """Given a query, retrieves the related documents.

        Args:
            query: The query string to use for document retrieval.

        Returns:
            List[Any]: A list of related documents.

        Raises:
            RagException: If an error occurs during document retrieval.
        """
        try:
            return self.retriever.invoke(query)
        except Exception as e:
            logger.error("Error retrieving related docs: %s", e)
            raise RagException(f"An issue occurred when retrieving the related docs: {str(e)} given the query: {query}")

    def retrieve_related_chunks_for_context(self, query: str) -> List[Any]:
        """Given a query, retrieves the related chunks for the context.

        Args:
            query: The query string to use for context retrieval.

        Returns:
            List[Any]: A list of related chunks.

        Raises:
            RagException: If an error occurs during chunk retrieval.
        """
        try:
            return self.chunks_retriever.invoke(query)
        except Exception as e:
            logger.error("Error retrieving related chunks: %s", e)
            raise RagException(f"An issue occurred when retrieving the related chunks: {str(e)} given the query: {query}")

    def invoke(self, query: str) -> Dict[str, Any]:
        """Given a query, retrieves related documents and provides them to the chain for LLM processing.

        Args:
            query: The query string to process.

        Returns:
            Dict[str, Any]: A dictionary containing the LLM output and related documents.

        Raises:
            RagException: If an error occurs during invocation.
        """
        try:
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
            llm_output = self.rag_chain.invoke(context)
            logger.info("LLM has provided an answer")
            logger.debug("LLM output: %s", llm_output)

            # Return the LLM output along with the related documents
            return {
                "llm_output": llm_output,
                "related_documents": related_docs
            }
        except Exception as e:
            logger.error("Error during invocation: %s", e)
            raise RagException(f"An issue occurred during the invocation: {str(e)}")
