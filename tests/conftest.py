import pytest
import os
import sys
from types import SimpleNamespace

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import config
except ImportError:
    config = None

@pytest.fixture(scope="session")
def diagnosis_only_test_file_path():
    """Returns the path to the test data for 'diagnosis_only' mode from config."""
    if not config or not hasattr(config, 'TEST_DIAGNOSIS_ONLY_DATA_PATH'):
        pytest.skip("TEST_DIAGNOSIS_ONLY_DATA_PATH not found in config.py")
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    file_path = os.path.join(project_root, config.TEST_DIAGNOSIS_ONLY_DATA_PATH)
    
    if not os.path.exists(file_path):
        pytest.skip(f"Test data file not found at: {file_path}")
        
    return file_path

@pytest.fixture(scope="session")
def multi_entity_test_file_path():
    """Returns the path to the test data for 'multi_entity' mode from config."""
    if not config or not hasattr(config, 'TEST_MULTI_ENTITY_DATA_PATH'):
        pytest.skip("TEST_MULTI_ENTITY_DATA_PATH not found in config.py")
        
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    file_path = os.path.join(project_root, config.TEST_MULTI_ENTITY_DATA_PATH)
    
    if not os.path.exists(file_path):
        pytest.skip(f"Test data file not found at: {file_path}")
        
    return file_path

@pytest.fixture
def diagnosis_only_config(diagnosis_only_test_file_path):
    """Returns a config object for testing 'diagnosis_only' mode."""
    if not config:
        pytest.skip("Main config.py could not be imported.")
        
    return SimpleNamespace(
        DATA_PATH=diagnosis_only_test_file_path,
        ENTITY_MODE='diagnosis_only',
        TEXT_COLUMN=config.TEXT_COLUMN,
        PATIENT_ID_COLUMN=config.PATIENT_ID_COLUMN,
        DIAGNOSES_COLUMN=config.DIAGNOSES_COLUMN,
        DATES_COLUMN=config.DATES_COLUMN,
        RELATIONSHIP_GOLD_COLUMN=config.RELATIONSHIP_GOLD_COLUMN,
        TRAINING_SET_RATIO=config.TRAINING_SET_RATIO,
        DATA_SPLIT_RANDOM_SEED=config.DATA_SPLIT_RANDOM_SEED
    )

@pytest.fixture
def multi_entity_config(multi_entity_test_file_path):
    """Returns a config object for testing 'multi_entity' mode."""
    if not config:
        pytest.skip("Main config.py could not be imported.")

    return SimpleNamespace(
        DATA_PATH=multi_entity_test_file_path,
        ENTITY_MODE='multi_entity',
        TEXT_COLUMN=config.TEXT_COLUMN,
        PATIENT_ID_COLUMN=config.PATIENT_ID_COLUMN,
        SNOMED_COLUMN=config.SNOMED_COLUMN,
        UMLS_COLUMN=config.UMLS_COLUMN,
        DATES_COLUMN=config.DATES_COLUMN,
        ENTITY_GOLD_COLUMN=config.ENTITY_GOLD_COLUMN,
        RELATIONSHIP_GOLD_COLUMN=config.RELATIONSHIP_GOLD_COLUMN,
        TRAINING_SET_RATIO=config.TRAINING_SET_RATIO,
        DATA_SPLIT_RANDOM_SEED=config.DATA_SPLIT_RANDOM_SEED
    ) 