import pytest
from unittest.mock import patch, MagicMock
from extractors.extractor_factory import create_extractor
from extractors.base_extractor import BaseRelationExtractor
from extractors.naive_extractor import NaiveExtractor
from extractors.custom_extractor import CustomExtractor
from extractors.bert_extractor import BertExtractor
from extractors.openai_extractor import OpenAIExtractor
from extractors.llama_extractor import LlamaExtractor
from extractors.relcat_extractor import RelcatExtractor

class TestExtractorFactory:
    def test_create_naive_extractor(self):
        """Test creating a naive extractor."""
        with patch('extractors.naive_extractor.NaiveExtractor') as mock_extractor:
            mock_instance = MagicMock()
            mock_extractor.return_value = mock_instance
            
            config = MagicMock()
            config.PROXIMITY_MAX_DISTANCE = 200
            
            extractor = create_extractor('naive', config)
            
            # Check that the correct extractor was created
            mock_extractor.assert_called_once()
            assert extractor == mock_instance
    
    def test_create_custom_extractor(self):
        """Test creating a custom extractor."""
        with patch('extractors.custom_extractor.CustomExtractor') as mock_extractor:
            mock_instance = MagicMock()
            mock_extractor.return_value = mock_instance
            
            config = MagicMock()
            config.MODEL_PATH = 'path/to/model'
            config.VOCAB_PATH = 'path/to/vocab'
            config.PREDICTION_MAX_DISTANCE = 500
            config.PREDICTION_MAX_CONTEXT_LEN = 512
            config.CUSTOM_CONFIDENCE_THRESHOLD = 0.5
            config.DEVICE = 'cpu'
            
            extractor = create_extractor('custom', config)
            
            # Check that the correct extractor was created
            mock_extractor.assert_called_once()
            assert extractor == mock_instance
    
    def test_create_bert_extractor(self):
        """Test creating a BERT extractor."""
        with patch('extractors.bert_extractor.BertExtractor') as mock_extractor:
            mock_instance = MagicMock()
            mock_extractor.return_value = mock_instance
            
            config = MagicMock()
            config.BERT_MODEL_PATH = 'path/to/bert'
            config.BERT_CONFIDENCE_THRESHOLD = 0.185
            config.DEVICE = 'cpu'
            
            extractor = create_extractor('bert', config)
            
            # Check that the correct extractor was created
            mock_extractor.assert_called_once()
            assert extractor == mock_instance
    
    def test_create_openai_extractor(self):
        """Test creating an OpenAI extractor."""
        with patch('extractors.openai_extractor.OpenAIExtractor') as mock_extractor:
            mock_instance = MagicMock()
            mock_extractor.return_value = mock_instance
            
            config = MagicMock()
            config.OPENAI_MODEL = 'gpt-4o'
            
            extractor = create_extractor('openai', config)
            
            # Check that the correct extractor was created
            mock_extractor.assert_called_once()
            assert extractor == mock_instance
    
    def test_create_llama_extractor(self):
        """Test creating a Llama extractor."""
        with patch('extractors.llama_extractor.LlamaExtractor') as mock_extractor:
            mock_instance = MagicMock()
            mock_extractor.return_value = mock_instance
            
            config = MagicMock()
            config.LLAMA_MODEL_PATH = 'path/to/llama'
            
            extractor = create_extractor('llama', config)
            
            # Check that the correct extractor was created
            mock_extractor.assert_called_once()
            assert extractor == mock_instance
    
    def test_create_relcat_extractor(self):
        """Test creating a RelCAT extractor."""
        with patch('extractors.relcat_extractor.RelcatExtractor') as mock_extractor:
            mock_instance = MagicMock()
            mock_extractor.return_value = mock_instance
            
            config = MagicMock()
            config.MEDCAT_MODEL_PATH = 'path/to/medcat/model'
            config.MEDCAT_CDB_PATH = 'path/to/medcat/cdb'
            
            extractor = create_extractor('relcat', config)
            
            # Check that the correct extractor was created
            mock_extractor.assert_called_once()
            assert extractor == mock_instance
    
    def test_create_unknown_extractor(self):
        """Test creating an unknown extractor type."""
        config = MagicMock()
        
        with pytest.raises(ValueError) as excinfo:
            create_extractor('unknown', config)
        
        assert "Unknown extractor type" in str(excinfo.value)
    
    def test_extractor_inheritance(self):
        """Test that all extractors inherit from BaseRelationExtractor."""
        assert issubclass(NaiveExtractor, BaseRelationExtractor)
        assert issubclass(CustomExtractor, BaseRelationExtractor)
        assert issubclass(BertExtractor, BaseRelationExtractor)
        assert issubclass(OpenAIExtractor, BaseRelationExtractor)
        assert issubclass(LlamaExtractor, BaseRelationExtractor)
        assert issubclass(RelcatExtractor, BaseRelationExtractor) 