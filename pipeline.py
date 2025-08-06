import time
import model
import loader
import retriever
import hyperparameters as hp
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter

def build_rag_pipeline(file_path):
    # Load documents
    print("Loading documents...")
    documents = loader.load_document(file_path)

    # Preprocess text
    documents = loader.preprocess_document(documents)

    # Split documents into chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=hp.CHUNK_SIZE,
        chunk_overlap=hp.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    # Filter out very small chunks (likely artifacts)
    chunks = [chunk for chunk in chunks if len(chunk.page_content.strip()) > 100]
    print(f"After filtering: {len(chunks)} chunks remain.")

    # Setup retrievers with smaller chunk sizes to avoid context overflow
    print("Setting up retrievers...")
    dense_retriever = retriever.setup_gpu_embeddings(chunks)
    sparse_retriever = BM25Retriever.from_documents(chunks)
    sparse_retriever.k = hp.RETRIEVAL_K

    hybrid_retriever = EnsembleRetriever(
        retrievers=[sparse_retriever, dense_retriever],
        weights=[hp.SPARSE_RATIO, 1-hp.SPARSE_RATIO]
    )

    # Try Ollama LLM setup
    print("\nSetting up Ollama LLM...")
    llm = model.setup_ollama_llm()

    if llm is None:
        raise RuntimeError("No LLM could be initialized.")

    return hybrid_retriever, None

def format_docs(docs):
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

def generate_response_with_progress(query, retriever, llm):
    print(f"\nProcessing query: '{query}'")

    start_time = time.time()

    print("📚 Retrieving relevant documents...", end='', flush=True)
    retrieved_docs = retriever.invoke(query)
    print(f" Found {len(retrieved_docs)} documents")

    print("Preparing context...", end='', flush=True)
    context = format_docs(retrieved_docs)
    print(" Context ready")

    # Prompt template
    formatted_prompt = f"""You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.

Context:
{context}

Question: {query}

Answer:"""

    print("\n" + "="*50)
    print("AI RESPONSE (Streaming)")
    print("="*50)

    full_response = ""
    word_count = 0

    try:
        for i, token in enumerate(llm.generate_stream(formatted_prompt)):
            if i == 0:
                print("\rThinking...  \n", end='', flush=True)
            print(token, end='', flush=True)
            full_response += token

            # Count words for progress
            if token.strip():
                word_count += len(token.split())

        print(f"\n\n📊 Generated {word_count} words")

    except Exception as e:
        print(f"\nStreaming error: {e}")
        print("Falling back to standard generation...")

        # Fallback to non-streaming call
        full_response = llm._call(formatted_prompt)
        print(full_response)

    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f}s")

    return full_response, retrieved_docs

def generate_response(query, retriever, llm):
    print(f"\nProcessing query: '{query}'")
    print("Retrieving relevant documents...")

    start_time = time.time()

    retrieved_docs = retriever.invoke(query)
    retrieval_time = time.time() - start_time
    print(f"Retrieved {len(retrieved_docs)} documents in {retrieval_time:.2f}s")

    context = format_docs(retrieved_docs)

    formatted_prompt = f"""You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.

Context:
{context}

Question: {query}

You must give your final answer using 'Final Answer:' prefix"""

    print("\n" + "="*50)
    print("ANSWER (NON-STREAMING)")
    print("="*50)

    # Use the non-streaming _call method for fallback or non-streaming mode
    full_response = llm._call(formatted_prompt)
    print(full_response)

    generation_time = time.time() - start_time - retrieval_time
    total_time = time.time() - start_time

    print(f"\n\nGeneration time: {generation_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")

    return full_response, retrieved_docs

def interactive_streaming_loop(retriever, llm):
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
            user_query = input(f"\nEnter your query: ")

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
                _, source_docs = generate_response_with_progress(user_query, retriever, llm)
            else:
                _, source_docs = generate_response(user_query, retriever, llm)

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
            import traceback
            traceback.print_exc()
            print("Please try again or type 'quit' to exit.")