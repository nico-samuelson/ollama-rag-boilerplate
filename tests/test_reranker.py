import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import Document
from typing import Dict, List
from reranker import RAGReranker


class TestRAGReranker:
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Mock tokenizer for testing"""
        tokenizer = Mock()
        tokenizer.convert_tokens_to_ids.side_effect = lambda token: {"no": 1, "yes": 2}[token]
        tokenizer.encode.side_effect = lambda text, add_special_tokens=False: [1, 2, 3] if "prefix" in text else [4, 5, 6]
        tokenizer.pad = Mock(return_value={
            'input_ids': torch.tensor([[1, 2, 3, 4, 5, 0], [1, 2, 3, 4, 5, 6]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])
        })
        return tokenizer

    @pytest.fixture
    def mock_model(self):
        """Mock model for testing"""
        model = Mock()
        model.device = 'mps'
        model.eval.return_value = model
        # Mock logits output
        mock_output = Mock()
        mock_output.logits = torch.tensor([[[0.1, 0.8, 0.3]], [[0.2, 0.6, 0.4]]])
        model.return_value = mock_output
        return model

    @pytest.fixture
    def reranker(self, mock_tokenizer, mock_model):
        """Create a RAGReranker instance with mocked dependencies"""
        with patch('reranker.AutoTokenizer.from_pretrained', return_value=mock_tokenizer), \
             patch('reranker.AutoModelForCausalLM.from_pretrained', return_value=mock_model), \
             patch('reranker.hp.RERANKER_NAME', 'test-model'):
            
            reranker = RAGReranker()
            return reranker

    def test_init_with_default_model(self, mock_tokenizer, mock_model):
        """Test initialization with default model name"""
        with patch('reranker.AutoTokenizer.from_pretrained', return_value=mock_tokenizer) as mock_tok_init, \
             patch('reranker.AutoModelForCausalLM.from_pretrained', return_value=mock_model) as mock_model_init, \
             patch('reranker.hp.RERANKER_NAME', 'default-model'):
            
            reranker = RAGReranker(model_name='default-model')
            
            # Check model loading was called
            mock_tok_init.assert_called_once_with('default-model', padding_side='left')
            mock_model_init.assert_called_once_with(
                'default-model',
                dtype=torch.float16,
                device_map="mps"
            )
            
            # Check attributes
            assert reranker.model_name == 'default-model'
            assert reranker.token_false_id == 1
            assert reranker.token_true_id == 2
            assert reranker.max_length == 16000

    def test_init_with_custom_model(self, mock_tokenizer, mock_model):
        """Test initialization with custom model name"""
        with patch('reranker.AutoTokenizer.from_pretrained', return_value=mock_tokenizer), \
             patch('reranker.AutoModelForCausalLM.from_pretrained', return_value=mock_model):
            
            reranker = RAGReranker(model_name='custom-model')
            assert reranker.model_name == 'custom-model'

    def test_format_instruction_with_instruction(self, reranker):
        """Test format_instruction method with provided instruction"""
        instruction = "Custom instruction for testing"
        query = "test query"
        doc_content = "test document content"
        
        result = reranker.format_instruction(instruction, query, doc_content)
        
        expected = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc_content}"
        assert result == expected

    def test_format_instruction_without_instruction(self, reranker):
        """Test format_instruction method with None instruction (uses default)"""
        query = "test query"
        doc_content = "test document content"
        
        result = reranker.format_instruction(None, query, doc_content)
        
        expected = f"<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: {query}\n<Document>: {doc_content}"
        assert result == expected

    def test_process_inputs(self, reranker):
        """Test process_inputs method"""
        pairs = ["test pair 1", "test pair 2"]
        
        # Mock the tokenizer call
        reranker.tokenizer.return_value = {
            'input_ids': [[7, 8, 9], [10, 11, 12]],
            'attention_mask': [[1, 1, 1], [1, 1, 1]]
        }
        
        result = reranker.process_inputs(pairs)
        
        # Check that tokenizer was called correctly
        reranker.tokenizer.assert_called_once_with(
            pairs,
            padding=False,
            truncation='longest_first',
            return_attention_mask=False,
            max_length=reranker.max_length - len(reranker.prefix_tokens) - len(reranker.suffix_tokens)
        )
        
        # Check that pad was called
        assert reranker.tokenizer.pad.called

    def test_compute_logits(self, reranker):
        """Test compute_logits method"""
        # Create mock inputs
        inputs = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        }
        
        # Mock model output with proper logits shape
        mock_output = Mock()
        # Create logits tensor: [batch_size, sequence_length, vocab_size]
        logits = torch.zeros(1, 5, 10)  # vocab_size = 10
        logits[0, -1, 1] = -1.0  # false token logit
        logits[0, -1, 2] = 1.0   # true token logit
        mock_output.logits = logits
        reranker.model.return_value = mock_output
        
        scores = reranker.compute_logits(inputs)
        
        # Check that model was called
        reranker.model.assert_called_once_with(**inputs)
        
        # Check that we got a list of scores
        assert isinstance(scores, list)
        assert len(scores) == 1
        assert isinstance(scores[0], float)

    def test_rerank_with_empty_docs(self, reranker):
        """Test rerank method with empty document list"""
        query = "test query"
        docs = []
        
        reranked_docs, scores = reranker.rerank(query, docs)
        
        assert reranked_docs == []
        assert scores == []

    def test_rerank_with_string_docs(self, reranker):
        """Test rerank method with string documents"""
        query = "test query"
        docs = ["doc 1 content", "doc 2 content"]
        
        # Mock the methods that will be called
        with patch.object(reranker, 'process_inputs') as mock_process, \
             patch.object(reranker, 'compute_logits') as mock_compute:
            
            mock_process.return_value = {'input_ids': torch.tensor([[1, 2, 3]])}
            mock_compute.return_value = [0.8, 0.6]
            
            reranked_docs, scores = reranker.rerank(query, docs)
            
            # Check that methods were called
            assert mock_process.called
            assert mock_compute.called
            
            # Check results
            assert len(reranked_docs) == 2
            assert len(scores) == 2
            # Higher score should come first
            assert scores[0] >= scores[1]

    def test_rerank_with_langchain_documents(self, reranker):
        """Test rerank method with LangChain Document objects"""
        query = "test query"
        docs = [
            Document(page_content="doc 1 content", metadata={"source": "source1"}),
            Document(page_content="doc 2 content", metadata={"source": "source2"})
        ]
        
        with patch.object(reranker, 'process_inputs') as mock_process, \
             patch.object(reranker, 'compute_logits') as mock_compute:
            
            mock_process.return_value = {'input_ids': torch.tensor([[1, 2, 3]])}
            mock_compute.return_value = [0.9, 0.7]
            
            reranked_docs, scores = reranker.rerank(query, docs)
            
            # Check that document content was extracted correctly
            expected_pairs = [
                reranker.format_instruction('Given a user query, retrieve relevant passages that answer the query', 
                                          query, "doc 1 content"),
                reranker.format_instruction('Given a user query, retrieve relevant passages that answer the query', 
                                          query, "doc 2 content")
            ]
            
            # Verify process_inputs was called with formatted pairs
            mock_process.assert_called_once()
            call_args = mock_process.call_args[0][0]
            assert len(call_args) == 2

    def test_rerank_with_different_document_types(self, reranker):
        """Test rerank with various document object types"""
        query = "test query"
        
        # Create mock objects with different content attributes
        doc_with_content = Mock()
        doc_with_content.content = "content attribute"
        
        doc_with_text = Mock()
        doc_with_text.text = "text attribute"
        
        docs = [
            Document(page_content="page_content attribute"),
            doc_with_content,
            doc_with_text,
            "string document"
        ]
        
        with patch.object(reranker, 'process_inputs') as mock_process, \
             patch.object(reranker, 'compute_logits') as mock_compute:
            
            mock_process.return_value = {'input_ids': torch.tensor([[1, 2, 3]])}
            mock_compute.return_value = [0.9, 0.8, 0.7, 0.6]
            
            reranked_docs, scores = reranker.rerank(query, docs)
            
            # Verify all document types were processed
            assert len(reranked_docs) == 4
            assert len(scores) == 4

    def test_rerank_with_top_k(self, reranker):
        """Test rerank method with top_k parameter"""
        query = "test query"
        docs = ["doc 1", "doc 2", "doc 3", "doc 4"]
        top_k = 2
        
        with patch.object(reranker, 'process_inputs') as mock_process, \
             patch.object(reranker, 'compute_logits') as mock_compute:
            
            mock_process.return_value = {'input_ids': torch.tensor([[1, 2, 3]])}
            mock_compute.return_value = [0.5, 0.9, 0.3, 0.7]  # Second doc should be first, fourth second
            
            reranked_docs, scores = reranker.rerank(query, docs, top_k=top_k)
            
            # Check that only top_k documents are returned
            assert len(reranked_docs) == top_k
            assert len(scores) == top_k
            # Check ordering (highest scores first)
            assert scores[0] >= scores[1]

    def test_rerank_with_custom_instruction(self, reranker):
        """Test rerank method with custom instruction"""
        query = "test query"
        docs = ["doc 1"]
        custom_instruction = "Custom reranking instruction"
        
        with patch.object(reranker, 'process_inputs') as mock_process, \
             patch.object(reranker, 'compute_logits') as mock_compute:
            
            mock_process.return_value = {'input_ids': torch.tensor([[1, 2, 3]])}
            mock_compute.return_value = [0.8]
            
            reranker.rerank(query, docs, instruction=custom_instruction)
            
            # Verify the custom instruction was used
            call_args = mock_process.call_args[0][0]
            assert custom_instruction in call_args[0]

    def test_rerank_score_ordering(self, reranker):
        """Test that rerank properly orders documents by score"""
        query = "test query"
        docs = ["low score doc", "high score doc", "medium score doc"]
        
        with patch.object(reranker, 'process_inputs') as mock_process, \
             patch.object(reranker, 'compute_logits') as mock_compute:
            
            mock_process.return_value = {'input_ids': torch.tensor([[1, 2, 3]])}
            # Return scores in order: low, high, medium
            mock_compute.return_value = [0.2, 0.9, 0.6]
            
            reranked_docs, scores = reranker.rerank(query, docs)
            
            # Check that documents are ordered by score (descending)
            assert reranked_docs[0] == "high score doc"
            assert reranked_docs[1] == "medium score doc"
            assert reranked_docs[2] == "low score doc"
            assert scores == [0.9, 0.6, 0.2]

    @pytest.mark.parametrize("top_k", [None, 1, 3, 10])
    def test_rerank_top_k_variations(self, reranker, top_k):
        """Test rerank with different top_k values"""
        query = "test query"
        docs = ["doc1", "doc2", "doc3"]
        
        with patch.object(reranker, 'process_inputs') as mock_process, \
             patch.object(reranker, 'compute_logits') as mock_compute:
            
            mock_process.return_value = {'input_ids': torch.tensor([[1, 2, 3]])}
            mock_compute.return_value = [0.8, 0.6, 0.9]
            
            reranked_docs, scores = reranker.rerank(query, docs, top_k=top_k)
            
            expected_length = min(top_k, len(docs)) if top_k is not None else len(docs)
            assert len(reranked_docs) == expected_length
            assert len(scores) == expected_length

# Integration-style tests (still using mocks but testing the full flow)
class TestRAGRerankerIntegration:
    def test_full_rerank_pipeline(self):
        """Test the complete rerank pipeline with realistic mocks"""
        with patch('reranker.AutoTokenizer.from_pretrained') as mock_tok_init, \
             patch('reranker.AutoModelForCausalLM.from_pretrained') as mock_model_init, \
             patch('reranker.hp.RERANKER_NAME', 'test-model'):
            
            # Setup tokenizer mock
            mock_tokenizer = Mock()
            mock_tokenizer.convert_tokens_to_ids.side_effect = lambda token: {"no": 100, "yes": 200}[token]
            mock_tokenizer.encode.side_effect = lambda text, **kwargs: list(range(len(text) // 10))
            mock_tokenizer.return_value = {
                'input_ids': [[1, 2, 3, 4], [5, 6, 7, 8]],
                'attention_mask': [[1, 1, 1, 1], [1, 1, 1, 1]]
            }
            mock_tokenizer.pad.return_value = {
                'input_ids': torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
                'attention_mask': torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]])
            }
            mock_tok_init.return_value = mock_tokenizer
            
            # Setup model mock
            mock_model = Mock()
            mock_model.device = 'mps'
            mock_model.eval.return_value = mock_model
            
            # Create realistic logits output
            batch_size = 2
            vocab_size = 1000
            seq_len = 4
            logits = torch.randn(batch_size, seq_len, vocab_size)
            logits[:, -1, 100] = torch.tensor([-2.0, -1.0])  # "no" token logits
            logits[:, -1, 200] = torch.tensor([1.0, 2.0])    # "yes" token logits
            
            mock_output = Mock()
            mock_output.logits = logits
            mock_model.return_value = mock_output
            mock_model_init.return_value = mock_model
            
            # Initialize reranker
            reranker = RAGReranker()
            
            # Test with real document objects
            query = "What is machine learning?"
            docs = [
                Document(page_content="Machine learning is a subset of AI", metadata={"id": 1}),
                Document(page_content="Python is a programming language", metadata={"id": 2})
            ]
            
            # Run reranking
            reranked_docs, scores = reranker.rerank(query, docs, top_k=2)
            
            # Verify results
            assert len(reranked_docs) == 2
            assert len(scores) == 2
            assert all(isinstance(doc, Document) for doc in reranked_docs)
            assert all(isinstance(score, float) for score in scores)
            # Scores should be in descending order
            assert scores[0] >= scores[1]