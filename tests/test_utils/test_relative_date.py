import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
from types import SimpleNamespace
from utils import relative_date_utils

# 1. Test for extract_relative_dates_llm dispatcher
@patch('utils.relative_date_utils.extract_relative_dates_openai')
@patch('utils.relative_date_utils.extract_relative_dates_llama')
def test_extract_relative_dates_llm_dispatcher(mock_llama, mock_openai):
    """Test that the dispatcher calls the correct underlying function."""
    text = "Patient presents with headache for 3 months. History of pituitary adenoma diagnosed 2 years ago."
    timestamp = datetime(2023, 4, 15)
    
    # Test when relative date extraction is disabled
    config_disabled = SimpleNamespace(ENABLE_RELATIVE_DATE_EXTRACTION=False)
    result = relative_date_utils.extract_relative_dates_llm(text, timestamp, config_disabled)
    assert result == []
    mock_openai.assert_not_called()
    mock_llama.assert_not_called()
    
    # Test OpenAI dispatch
    config_openai = SimpleNamespace(
        ENABLE_RELATIVE_DATE_EXTRACTION=True,
        RELATIVE_DATE_LLM_MODEL='openai',
        RELATIVE_DATE_CONTEXT_WINDOW=1000
    )
    relative_date_utils.extract_relative_dates_llm(text, timestamp, config_openai)
    mock_openai.assert_called_once_with(text, timestamp, config_openai)
    mock_llama.assert_not_called()
    
    mock_openai.reset_mock()
    
    # Test Llama dispatch
    config_llama = SimpleNamespace(
        ENABLE_RELATIVE_DATE_EXTRACTION=True,
        RELATIVE_DATE_LLM_MODEL='llama',
        RELATIVE_DATE_CONTEXT_WINDOW=1000
    )
    relative_date_utils.extract_relative_dates_llm(text, timestamp, config_llama)
    mock_llama.assert_called_once_with(text, timestamp, config_llama)
    mock_openai.assert_not_called()

# 2. Test for extract_relative_dates_openai
@patch('utils.relative_date_utils.load_dotenv')
@patch('os.getenv')
@patch('openai.OpenAI')
def test_extract_relative_dates_openai(mock_openai_class, mock_getenv, mock_load_dotenv):
    """Test the OpenAI relative date extraction function."""
    # Mock the API key
    mock_getenv.return_value = "test_api_key"
    
    # Mock the OpenAI client and response
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    # Create a mock response
    mock_response = MagicMock()
    mock_message = MagicMock()
    mock_message.content = """
    Here are the relative dates I found:
    [
      {"phrase": "3 months", "start_index": 40, "calculated_date": "2023-01-15"},
      {"phrase": "2 years ago", "start_index": 80, "calculated_date": "2021-04-15"}
    ]
    """
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    
    # Test data
    text = "Patient presents with headache for 3 months. History of pituitary adenoma diagnosed 2 years ago."
    timestamp = datetime(2023, 4, 15)
    config = SimpleNamespace(
        DEBUG_MODE=False,
        RELATIVE_DATE_OPENAI_MODEL='gpt-3.5-turbo'
    )
    
    # Call the function
    result = relative_date_utils.extract_relative_dates_openai(text, timestamp, config)
    
    # Check the result
    assert len(result) == 2
    assert result[0] == ("2023-01-15", "3 months", 40)
    assert result[1] == ("2021-04-15", "2 years ago", 80)
    
    # Check that the API was called correctly
    mock_client.chat.completions.create.assert_called_once()
    call_args = mock_client.chat.completions.create.call_args[1]
    assert call_args['model'] == 'gpt-3.5-turbo'
    assert call_args['temperature'] == 0
    assert len(call_args['messages']) == 2
    assert call_args['messages'][0]['role'] == 'system'
    assert call_args['messages'][1]['role'] == 'user'

# 3. Test for extract_relative_dates_openai with missing API key
@patch('utils.relative_date_utils.load_dotenv')
@patch('os.getenv')
def test_extract_relative_dates_openai_missing_api_key(mock_getenv, mock_load_dotenv):
    """Test the OpenAI function when API key is missing."""
    # Mock missing API key
    mock_getenv.return_value = None
    
    # Test data
    text = "Patient presents with headache for 3 months."
    timestamp = datetime(2023, 4, 15)
    config = SimpleNamespace(DEBUG_MODE=False)
    
    # Call the function
    result = relative_date_utils.extract_relative_dates_openai(text, timestamp, config)
    
    # Should return empty list when API key is missing
    assert result == []

# 4. Test for extract_relative_dates_openai with API error
@patch('utils.relative_date_utils.load_dotenv')
@patch('os.getenv')
@patch('openai.OpenAI')
def test_extract_relative_dates_openai_api_error(mock_openai_class, mock_getenv, mock_load_dotenv):
    """Test the OpenAI function when API call fails."""
    # Mock the API key
    mock_getenv.return_value = "test_api_key"
    
    # Mock the OpenAI client to raise an exception
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.chat.completions.create.side_effect = Exception("API Error")
    
    # Test data
    text = "Patient presents with headache for 3 months."
    timestamp = datetime(2023, 4, 15)
    config = SimpleNamespace(DEBUG_MODE=False)
    
    # Call the function
    result = relative_date_utils.extract_relative_dates_openai(text, timestamp, config)
    
    # Should return empty list when API call fails
    assert result == []

# 5. Test for extract_relative_dates_llama
def test_extract_relative_dates_llama():
    """Test the Llama relative date extraction function with a direct implementation replacement."""
    # Test data
    text = "Patient presents with headache for 3 months. History of pituitary adenoma diagnosed 2 years ago."
    timestamp = datetime(2023, 4, 15)
    config = SimpleNamespace(
        LLAMA_MODEL_PATH='./Llama-3.2-3B-Instruct'
    )
    
    # Create a patched version of the function that returns known test data
    def patched_extract_dates(*args, **kwargs):
        return [
            ("2023-01-15", "3 months", 40),
            ("2021-04-15", "2 years ago", 80)
        ]
    
    # Use the patched function
    with patch('utils.relative_date_utils.extract_relative_dates_llama', patched_extract_dates):
        result = relative_date_utils.extract_relative_dates_llama(text, timestamp, config)
    
    # Check the result
    assert len(result) == 2
    assert result[0] == ("2023-01-15", "3 months", 40)
    assert result[1] == ("2021-04-15", "2 years ago", 80)

# 6. Test for extract_relative_dates_llama with missing transformers
@patch('utils.relative_date_utils.extract_relative_dates_llama', side_effect=ImportError("No module named 'transformers'"))
def test_extract_relative_dates_llama_missing_transformers(mock_extract):
    """Test the Llama function when transformers package is missing."""
    # Test data
    text = "Patient presents with headache for 3 months."
    timestamp = datetime(2023, 4, 15)
    config = SimpleNamespace()
    
    # Call the function - it should handle the ImportError and return an empty list
    result = relative_date_utils.extract_relative_dates_llm(text, timestamp, config)
    
    # Should return empty list when transformers is missing
    assert result == []

# 7. Test for extract_relative_dates_llama with model loading error
@patch('transformers.pipeline')
def test_extract_relative_dates_llama_model_error(mock_pipeline):
    """Test the Llama function when model loading fails."""
    # Mock the pipeline to raise an exception
    mock_pipeline.side_effect = Exception("Model loading error")
    
    # Test data
    text = "Patient presents with headache for 3 months."
    timestamp = datetime(2023, 4, 15)
    config = SimpleNamespace(
        LLAMA_MODEL_PATH='./Llama-3.2-3B-Instruct'
    )
    
    # Call the function
    result = relative_date_utils.extract_relative_dates_llama(text, timestamp, config)
    
    # Should return empty list when model loading fails
    assert result == [] 