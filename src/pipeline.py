import time
import traceback
from model import OllamaLLM
import hyperparameters as hp
from typing import List, Tuple
from reranker import RAGReranker
from loader import DocumentLoader
from langchain.schema import Document
from retriever import HybridRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RAGPipeline:
    def __init__(self,
                 file_path: str,
                 chunk_size: int = hp.CHUNK_SIZE,
                 chunk_overlap: int = hp.CHUNK_OVERLAP,
                 reranker_name: str = hp.RERANKER_NAME,
                 model_name: str = hp.MODEL_NAME,
                 temperature: float = 0.5,
                 max_new_tokens: int = 1024,
    ):
        self.file_path = file_path
        self.chunks = []
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.reranker_name = reranker_name
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        self.loader = DocumentLoader(self.file_path)
        self.retriever = HybridRetriever(self.chunks)
        self.reranker = RAGReranker(self.reranker_name)
        self.llm = OllamaLLM(
            model=self.model_name,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
        )

        self.load_and_preprocess_document()

    def load_and_preprocess_document(self):
        """Load and preprocess the document from file path."""

        document = self.loader.load_document(self.file_path)
        document = self.loader.preprocess_document(document)
        self.split_document(document)

    def split_document(self, document: List[Document]):
        """Split the document into chunks."""

        # Split documents into chunks
        print("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(document)
        print(f"Split into {len(chunks)} chunks.")

        # Filter out very small chunks (likely artifacts)
        self.chunks = [chunk for chunk in chunks if len(chunk.page_content.strip()) > 100]
        print(f"After filtering: {len(self.chunks)} chunks remain.")

    def format_docs(self, docs: List[Document]) -> str:
        """Convert list of documents into a single string for prompt context."""
        
        formatted = []
        total_chars = 0

        for i, doc in enumerate(docs):
            source_info = ""
            if hasattr(doc, 'metadata') and doc.metadata:
                page = doc.metadata.get('page', 'Unknown')
                source_info = f" (Page {page})"

            # Format the document
            doc_text = f"Source {i+1}{source_info}:\n{doc.page_content}"

            formatted.append(doc_text)
            total_chars += len(doc_text) + 2

        result = "\n\n".join(formatted)
        print(f"\nContext length: ~{len(result)} characters")
        return result

    def retrieve_relevant_docs(self, query: str) -> Tuple[List[Document], Tuple[List[Document], List[float]]]:
        """
        Retrieve and rerank documents based on the query.

        Args:
            query (str): The user query.
        Returns:
            tuple: (list of retrieved documents, list of reranked documents)
        """

        print("📚 Retrieving relevant documents...", end='', flush=True)
        retrieved_docs = self.retriever.as_retriever().invoke(query)
        print(f" Found {len(retrieved_docs)} documents")

        print("📚 Reranking documents...", end='', flush=True)
        reranked_docs = self.reranker.rerank(query, retrieved_docs, 5)
        print("Selected top 10 documents")

        return retrieved_docs, reranked_docs

    def prepare_prompt(self, query: str, docs: List[str]) -> str:
        """Inject the prompt with context and user query."""

        print("Preparing context...", end='', flush=True)
        context = self.format_docs(docs)
        print(" Context ready")

        # Prompt template
        formatted_prompt = f"""You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question. Don't say according to source 1, passage 1, etc. in your answer.
        If you don't know the answer, just say that you don't know.

        Context:
        {context}

        Question: {query}"""

        return formatted_prompt

    def generate_response_with_progress(self, query: str) -> Tuple[str, List[Document]]:
        """
        Generate response with real-time streaming output.
        
        Args:
            query (str): The user query.
        Returns:
            tuple: (response string, list of source documents)
        """

        print(f"\nProcessing query: '{query}'")

        _, reranked_docs = self.retrieve_relevant_docs(query)
        prompt = self.prepare_prompt(query, reranked_docs[0])
        start_time = time.time()

        print("\n" + "="*50)
        print("AI RESPONSE (Streaming)")
        print("="*50)

        full_response = ""

        try:
            for i, token in enumerate(self.llm.generate_stream(prompt)):
                if i == 0:
                    print("\rThinking...  \n", end='', flush=True)
                print(token, end='', flush=True)
                full_response += token
        except Exception as e:
            print(f"\nStreaming error: {e}")
            print("Falling back to standard generation...")

            # Fallback to non-streaming call
            full_response = self.llm._call(prompt)
            print(full_response)

        total_time = time.time() - start_time
        print(f"\n\Total time: {total_time:.2f}s")

        return full_response, reranked_docs[0]

    def generate_response(self, query: str) -> Tuple[str, List[Document]]:
        """
        Generate response in standard (non-streaming) mode.
        
        Args:
            query (str): The user query.
        Returns:
            tuple: (response string, list of source documents)
        """

        print(f"\nProcessing query: '{query}'")

        _, reranked_docs = self.retrieve_relevant_docs(query)
        prompt = self.prepare_prompt(query, reranked_docs[0])
        start_time = time.time()

        print("\n" + "="*50)
        print("ANSWER (NON-STREAMING)")
        print("="*50)

        full_response = self.llm._call(prompt)
        print(full_response)

        total_time = time.time() - start_time
        print(f"\n\Total time: {total_time:.2f}s")

        return full_response, reranked_docs[0]

    def interactive_streaming_loop(self):
        """Interactive loop for user queries with streaming output."""

        print(f"\nRAG Pipeline ready")
        print("📚 Document successfully processed and indexed!")
        print("\nCommands:")
        print("  'quit' - Exit the program")
        print("  'sources' - Toggle source document display")
        print("  'stream' - Toggle between streaming/standard mode")

        show_sources = True
        use_streaming = True

        while True:
            try:
                print("\n" + "="*100)
                user_query = input(f"Enter your query: ")

                if user_query.lower() == 'quit':
                    break
                elif user_query.lower() == 'sources':
                    show_sources = not show_sources
                    print(f"Source documents {'enabled' if show_sources else 'disabled'}")
                    continue
                elif user_query.lower() == 'stream':
                    use_streaming = not use_streaming
                    print(f"Switched to {'streaming' if use_streaming else 'standard'} mode")
                    continue
                elif user_query.strip() == '':
                    continue

                # Generate response based on mode
                if use_streaming:
                    _, source_docs = self.generate_response_with_progress(user_query)
                else:
                    _, source_docs = self.generate_response(user_query)

                # Show sources if enabled
                if show_sources:
                    print("\n" + "="*50)
                    print(f"📚 SOURCE DOCUMENTS ({len(source_docs)} retrieved)")
                    print("="*50)
                    for i, doc in enumerate(source_docs):
                        page_info = ""
                        if hasattr(doc, 'metadata') and doc.metadata:
                            page = doc.metadata.get('page', 'Unknown')
                            page_info = f" (Page {page})"

                        print(f"\n📄 Source {i+1}{page_info}")
                        print("-" * 30)
                        content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                        print(content)
                        print("-" * 30)

            except KeyboardInterrupt:
                print("\n\nInterrupted by user")
                break
            except Exception as e:
                print(f"\nError: {e}")
                traceback.print_exc()
                print("Please try again or type 'quit' to exit.")