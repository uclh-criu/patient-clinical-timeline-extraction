import pytest
from unittest.mock import MagicMock
from extractors.base_extractor import BaseRelationExtractor

class TestBaseExtractor:
    def test_init(self):
        """Test BaseRelationExtractor initialization."""
        # Create a concrete subclass for testing
        class ConcreteExtractor(BaseRelationExtractor):
            def extract(self, text, entities=None, note_id=None, patient_id=None):
                return []
            
            def load(self):
                return True
        
        # Initialize the extractor
        extractor = ConcreteExtractor(name="TestExtractor")
        
        # Check that the name was set correctly
        assert extractor.name == "TestExtractor"
    
    def test_abstract_extract_method(self):
        """Test that the extract method is abstract and must be implemented by subclasses."""
        # Try to instantiate the abstract base class
        with pytest.raises(TypeError) as excinfo:
            extractor = BaseRelationExtractor(name="TestExtractor")
        
        # Check that the error message mentions the abstract method
        assert "abstract" in str(excinfo.value).lower() or "implement" in str(excinfo.value).lower()
    
    def test_extract_implementation(self):
        """Test a concrete implementation of the extract method."""
        # Create a concrete subclass for testing
        class ConcreteExtractor(BaseRelationExtractor):
            def extract(self, text, entities=None, note_id=None, patient_id=None):
                # Simple implementation that returns a fixed result
                return [
                    {
                        'entity_label': 'test_entity',
                        'entity_category': 'test_category',
                        'date': '2023-01-01',
                        'confidence': 0.9
                    }
                ]
            
            def load(self):
                return True
        
        # Initialize the extractor
        extractor = ConcreteExtractor(name="TestExtractor")
        
        # Call the extract method
        result = extractor.extract("Sample text", entities=None, note_id=1, patient_id="P001")
        
        # Check the result
        assert len(result) == 1
        assert result[0]['entity_label'] == 'test_entity'
        assert result[0]['entity_category'] == 'test_category'
        assert result[0]['date'] == '2023-01-01'
        assert result[0]['confidence'] == 0.9
    
    def test_extract_with_parameters(self):
        """Test that extract method parameters are correctly passed to the implementation."""
        # Create a mock implementation to verify parameter passing
        mock_extract = MagicMock(return_value=[])
        
        # Create a concrete subclass with a mock implementation
        class MockExtractor(BaseRelationExtractor):
            def extract(self, text, entities=None, note_id=None, patient_id=None):
                # Call the mock to record parameters
                return mock_extract(text, entities, note_id, patient_id)
            
            def load(self):
                return True
        
        # Initialize the extractor
        extractor = MockExtractor(name="MockExtractor")
        
        # Test data
        text = "Sample text"
        entities = (["entity1"], ["date1"])
        note_id = 42
        patient_id = "P042"
        
        # Call the extract method
        extractor.extract(text, entities=entities, note_id=note_id, patient_id=patient_id)
        
        # Verify that the mock was called with the correct parameters
        mock_extract.assert_called_once_with(text, entities, note_id, patient_id) 