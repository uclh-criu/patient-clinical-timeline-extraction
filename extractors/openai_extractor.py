import os
import json
from dotenv import load_dotenv
from extractors.base_extractor import BaseRelationExtractor

class OpenAIExtractor(BaseRelationExtractor):
    """
    Relation extractor that uses OpenAI's API to identify entity-date 
    relationships in clinical notes.
    
    Requires OPENAI_API_KEY to be set in a .env file in the project root.
    """
    
    def __init__(self, config):
        """
        Initialize the OpenAI extractor.
        
        Args:
            config: Configuration object or dict with necessary parameters.
        """
        self.config = config
        self.api_key = None 
        self.model_name = config.OPENAI_MODEL if hasattr(config, 'OPENAI_MODEL') else 'gpt-4o'
        self.name = f"OpenAI ({self.model_name})"
        self.client = None
        
        # Define category mappings
        self.category_mapping = {
            # SNOMED mappings
            "disorder": "diagnosis",
            "finding": "symptom",
            "procedure": "procedure",
            "product": "medication",
            
            # UMLS mappings
            "sign or symptom": "symptom",
            "finding": "symptom",
            "diagnostic procedure": "procedure",
            "therapeutic or preventive procedure": "procedure",
            "disease or syndrome": "diagnosis",
            "pharmacologic substance": "medication",
            
            # Default fallbacks
            "unknown": "symptom"
        }
        
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
            print("OpenAI Extractor: Client initialized successfully.")
            return True
        except Exception as e:
            print(f"Error setting up OpenAI client: {e}")
            return False
    
    def _map_category(self, category):
        """
        Map old category names to new simplified categories.
        
        Args:
            category (str): Original category name (lowercase).
            
        Returns:
            str: Mapped category name (diagnosis, symptom, procedure, or medication).
        """
        category = category.lower()
        return self.category_mapping.get(category, "symptom")
    
    def extract(self, text, entities=None):
        """
        Extract relationships using OpenAI's API.
        
        Args:
            text (str): The clinical note text.
            entities (tuple, optional): A tuple of (entities_list, dates) if already extracted.
            
        Returns:
            list: A list of dictionaries, each representing a relationship:
                {
                    'entity_label': str,     # The entity text
                    'entity_category': str,  # The entity category (diagnosis, symptom, procedure, medication)
                    'date': str,             # The date text
                    'confidence': float      # Model prediction confidence
                }
        """
        if self.client is None:
            print("OpenAI client not initialized. Call load() first.")
            return []
        
        # Entities are required for CSV data processing
        if entities is None:
            print("Error: entities parameter is required for CSV data processing.")
            return []
        
        entities_list, dates = entities
        
        # Process entities based on format
        entities_info = []
        for entity in entities_list:
            if isinstance(entity, dict):
                # New format: dict with label, start, category, etc.
                original_category = entity.get('category', 'unknown')
                if isinstance(original_category, list) and len(original_category) > 0:
                    original_category = original_category[0]
                
                mapped_category = self._map_category(original_category)
                
                entities_info.append({
                    "entity_label": entity.get('label', ''),
                    "entity_category": mapped_category,
                    "position": entity.get('start', 0)
                })
            else:
                # Legacy format: tuple of (label, position)
                entities_info.append({
                    "entity_label": entity[0],
                    "entity_category": "diagnosis",  # Default to diagnosis instead of disorder
                    "position": entity[1]
                })
        
        # Extract parsed date, raw date string, and position
        dates_info = [{"parsed_date": d[0], "raw_date": d[1], "position": d[2]} for d in dates]
        
        # Construct the prompt
        prompt = f"""

        You are tasked with doing relationship extraction between medical entities and dates from unstructured medical text.

        The full clinical note text is provided below. Additionally, lists of medical entities and dates, along with their character positions in the text, are provided.

        For each entity listed in 'Entities Info', identify the single most relevant date from the 'Dates Info' list based on the context in the 'Clinical Note'. 

        The entity categories have been simplified to one of four categories:
        - diagnosis: medical conditions, diseases, disorders
        - symptom: signs, symptoms, findings, complaints
        - procedure: diagnostic tests, surgeries, therapeutic procedures
        - medication: drugs, medications, pharmacologic substances

        Return ONLY a JSON array where each object represents a likely relationship and has the following structure:
        {{
            "entity_label": "name of entity from the Entities Info list",
            "entity_category": "category of the entity (diagnosis, symptom, procedure, or medication)",
            "date": "the parsed date (from parsed_date field) associated with the entity in YYYY-MM-DD format",
            "confidence": a number between 0 and 1 indicating your confidence in this association
        }}
        Do not include entities that have no associated date in the output.

        Clinical Note: {text}

        Entities Info: {entities_info}
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
                print(f"Error: Could not find JSON array in OpenAI response: {response_text}")
                return []
            
        except Exception as e:
            print(f"Error during OpenAI API call or processing: {e}")
            return [] 