import torch
import hyperparameters as hp
from langchain.schema import Document
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Union, Sequence, Optional

class RAGReranker:
    def __init__(self, model_name: str = hp.RERANKER_NAME):
        """Initialize the reranker with model loading done once"""
        self.model_name = model_name
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map=self._get_optimal_device()
        ).eval()

        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = 16000

        # Pre-compute prefix/suffix tokens
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

    def _get_optimal_device(self) -> str:
        """
        Automatically detect and return the best available device.
        Priority: CUDA > MPS > CPU
        """
        if torch.cuda.is_available():
            device = "cuda"
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            print("Using MPS (Apple Silicon) device")
        else:
            device = "cpu"
            print("Using CPU device")
        
        return device

    def format_instruction(self, instruction: Optional[str], query: str, doc_content: str) -> str:
        """Format the instruction for the reranker"""
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc_content}"

    def process_inputs(self, pairs: Sequence[str]) -> Dict[str, torch.Tensor]:
        """Process and tokenize input pairs"""
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation='longest_first',
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )

        # Add prefix and suffix tokens
        for i, token_ids in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + token_ids + self.suffix_tokens

        # Pad and convert to tensors
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)

        # Move to device
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)

        return inputs

    def compute_logits(self, inputs: Dict[str, torch.Tensor]) -> List[float]:
        """Compute relevance scores from model logits"""
        with torch.no_grad():
            batch_scores = self.model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()
        return scores

    def rerank(self,
               query: str,
               docs: Sequence[Union[str, Document]],
               top_k: int = 10,
               instruction: str = None
    ) -> Tuple[List[Document], List[float]]:
        """
        Rerank documents based on relevance to query

        Args:
            query (str): Single query string
            docs (list): List of documents (either strings or objects with text content)
            top_k (int): Number of top documents to return
            instruction (str): Custom instruction for reranking

        Returns:
            tuple: (reranked_docs, scores)
        """
        if not docs:
            return [], []

        # Handle different document formats
        doc_contents = []
        for doc in docs:
            if isinstance(doc, str):
                doc_contents.append(doc)
            elif hasattr(doc, 'page_content'):
                doc_contents.append(doc.page_content)
            elif hasattr(doc, 'content'):
                doc_contents.append(doc.content)
            elif hasattr(doc, 'text'):
                doc_contents.append(doc.text)
            else:
                doc_contents.append(str(doc))

        # Create instruction pairs - query is repeated for each document
        task = instruction or 'Given a user query, retrieve relevant passages that answer the query'
        pairs = [self.format_instruction(task, query, doc_content)
                for doc_content in doc_contents]

        # Process inputs and compute scores
        inputs = self.process_inputs(pairs)
        scores = self.compute_logits(inputs)

        # Sort by scores and apply top_k filtering
        doc_score_pairs = list(zip(docs, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None and top_k < len(docs):
            doc_score_pairs = doc_score_pairs[:top_k]

        reranked_docs = [pair[0] for pair in doc_score_pairs]
        final_scores = [pair[1] for pair in doc_score_pairs]

        return reranked_docs, final_scores