import os
import json
from dotenv import load_dotenv
from extractors.base_extractor import BaseRelationExtractor
from utils.extraction_utils import extract_entities

class LLMExtractor(BaseRelationExtractor):
    """
    Relation extractor that uses a Large Language Model (LLM) via OpenAI's API 
    to identify diagnosis-date relationships.
    
    Requires OPENAI_API_KEY to be set in a .env file in the project root.
    """
    
    def __init__(self, config):
        """
        Initialize the LLM extractor.
        
        Args:
            config: Configuration object or dict with necessary parameters.
        """
        self.config = config
        self.api_key = None 
        self.model_name = config.OPENAI_MODEL if hasattr(config, 'OPENAI_MODEL') else 'gpt-4o'
        self.name = f"LLM ({self.model_name})"
        self.client = None
        
    def load(self):
        """
        Load environment variables (including API key) and set up the OpenAI API client.
        
        Returns:
            bool: True if successfully loaded, False otherwise.
        """
        try:
            # Load environment variables from .env file
            load_dotenv()
            self.api_key = os.getenv('OPENAI_API_KEY')
            
            # Check if openai package is installed
            try:
                from openai import OpenAI
            except ImportError:
                print("Error: openai package not installed. Install with 'pip install openai'.")
                return False
            
            # Check if API key was loaded
            if not self.api_key:
                print("Error: OPENAI_API_KEY not found in .env file or environment variables.")
                return False
            
            # Initialize OpenAI client
            self.client = OpenAI(api_key=self.api_key)
            print("LLM Extractor: OpenAI client initialized successfully.")
            return True
        except Exception as e:
            print(f"Error setting up OpenAI client: {e}")
            return False
    
    def extract(self, text, entities=None):
        """
        Extract relationships using the configured LLM via OpenAI API.
        
        Args:
            text (str): The clinical note text.
            entities (tuple, optional): A tuple of (diagnoses, dates) if already extracted.
            
        Returns:
            list: A list of dictionaries, each representing a relationship:
                {
                    'diagnosis': str,      # The diagnosis text
                    'date': str,           # The date text
                    'confidence': float    # Model prediction confidence
                }
        """
        if self.client is None:
            print("LLM client (OpenAI) not initialized. Call load() first.")
            return []
        
        if entities is None:
            diagnoses, dates = extract_entities(text)
        else:
            diagnoses, dates = entities
        
        # Extract diagnosis names and positions
        diagnoses_info = [{"diagnosis": d[0], "position": d[1]} for d in diagnoses]
        # Extract parsed date, raw date string, and position
        dates_info = [{"parsed_date": d[0], "raw_date": d[1], "position": d[2]} for d in dates]
        
        # Construct the prompt
        prompt = f"""

        You are tasked with doing relationship extraction between diagnoses and dates from unstructured medical text.

        The full clinical note text is provided below. Additionally, lists of diagnoses and dates, along with their character positions in the text, are provided.

        For each diagnosis listed in 'Diagnoses Info', identify the single most relevant date from the 'Dates Info' list based on the context in the 'Clinical Note'. 

        Return ONLY a JSON array where each object represents a likely relationship and has the following structure:
        {{
            "diagnosis": "name of diagnosis from the Diagnoses Info list",
            "date": "the RAW date string (from raw_date field) associated with the diagnosis",
            "confidence": a number between 0 and 1 indicating your confidence in this association
        }}
        Do not include diagnoses that have no associated date in the output.

        Clinical Note: {text}

        Diagnoses Info: {diagnoses_info}
        Dates Info: {dates_info}

        Provide ONLY the JSON array, no other explanation.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a medical AI assistant specialized in analyzing unstructured clinical notes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,  
                max_tokens=2000
            )
            
            response_text = response.choices[0].message.content.strip()
            
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                relationships = json.loads(json_str)
                for rel in relationships:
                    if 'confidence' in rel:
                         try:
                             rel['confidence'] = float(rel['confidence'])
                         except (ValueError, TypeError):
                             rel['confidence'] = 0.0 
                    else:
                         rel['confidence'] = 1.0 
                return relationships
            else:
                print(f"Error: Could not find JSON array in LLM response: {response_text}")
                return []
            
        except Exception as e:
            print(f"Error during LLM API call or processing: {e}")
            return [] 