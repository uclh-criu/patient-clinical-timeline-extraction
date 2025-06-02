import os
import json
import torch
from extractors.base_extractor import BaseRelationExtractor
from tqdm import tqdm

class LlamaExtractor(BaseRelationExtractor):
    """
    Relation extractor that uses Llama 3.2 3B model locally to identify 
    diagnosis-date relationships in clinical notes.
    """
    
    def __init__(self, config):
        """
        Initialize the Llama extractor.
        
        Args:
            config: Configuration object or dict with necessary parameters.
        """
        self.config = config
        self.model_id = config.LLAMA_MODEL_PATH if hasattr(config, 'LLAMA_MODEL_PATH') else './Llama-3.2-3B-Instruct'
        self.name = "Llama-3.2"
        self.pipeline = None
        
    def load(self):
        """
        Load the Llama model and set up the pipeline.
        
        Returns:
            bool: True if successfully loaded, False otherwise.
        """
        try:
            # Check if transformers package is installed
            try:
                from transformers import pipeline
            except ImportError:
                print("Error: transformers package not installed. Install with 'pip install transformers'.")
                return False
                
            # Create a loading indicator since model loading can take time
            with tqdm(total=100, desc="Loading Llama model", unit="%") as pbar:
                print(f"Loading Llama model from {self.model_id}...")
                
                # Update progress to 25% - started loading
                pbar.update(25)
                
                # Load the model
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                
                # Update progress to 100% - done loading
                pbar.update(75)
                
            print("Llama Extractor: Model initialized successfully.")
            return True
        except Exception as e:
            print(f"Error setting up Llama model: {e}")
            return False
    
    def extract(self, text, entities=None):
        """
        Extract relationships using the Llama 3.2 model.
        
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
        if self.pipeline is None:
            print("Llama model not initialized. Call load() first.")
            return []
        
        # Entities are required for CSV data processing
        if entities is None:
            print("Error: entities parameter is required for CSV data processing.")
            return []
        
        diagnoses, dates = entities
        
        # Extract diagnosis names and positions
        diagnoses_info = [{"diagnosis": d[0], "position": d[1]} for d in diagnoses]
        # Extract parsed date, raw date string, and position
        dates_info = [{"parsed_date": d[0], "raw_date": d[1], "position": d[2]} for d in dates]
        
        # Construct the messages for the model
        system_prompt = "You are a medical AI assistant specialized in analyzing clinical notes to find relationships between diagnoses and dates."
        
        user_prompt = f"""
        I need you to identify relationships between diagnoses and dates in a clinical note.

        The full clinical note text is provided below, along with diagnoses and dates that have been extracted, including their positions in the text.
        
        For each diagnosis listed in 'Diagnoses Info', identify the single most relevant date from the 'Dates Info' list based on the context in the 'Clinical Note'.

        Return ONLY a JSON array where each object represents a likely relationship and has the following structure:
        {{
            "diagnosis": "name of diagnosis from the Diagnoses Info list",
            "date": "the parsed date (from parsed_date field) associated with the diagnosis in YYYY-MM-DD format",
            "confidence": a number between 0 and 1 indicating your confidence in this association
        }}
        Do not include diagnoses that have no associated date in the output.

        Clinical Note: {text}

        Diagnoses Info: {diagnoses_info}
        Dates Info: {dates_info}

        Provide ONLY the JSON array, no other explanation or text.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # Show a continuous spinner during inference
            import sys
            import time
            import threading
            
            # Create a threading event to signal when to stop the spinner
            stop_spinner = threading.Event()
            
            # Define the spinner function
            def spinner_function():
                chars = "|/-\\"
                i = 0
                start_time = time.time()
                while not stop_spinner.is_set():
                    elapsed = int(time.time() - start_time)
                    mins, secs = divmod(elapsed, 60)
                    timeformat = f"{mins:02d}:{secs:02d}"
                    sys.stdout.write(f"\rGenerating response... {chars[i % len(chars)]} [{timeformat} elapsed]")
                    sys.stdout.flush()
                    time.sleep(0.1)
                    i += 1
                
                # Clear the spinner line when done
                sys.stdout.write("\rInference complete!                                  \n")
                sys.stdout.flush()
            
            # Start the spinner in a separate thread
            spinner_thread = threading.Thread(target=spinner_function)
            spinner_thread.daemon = True
            spinner_thread.start()
            
            try:
                # Run the model
                outputs = self.pipeline(
                    messages,
                    max_new_tokens=2000,
                    do_sample=False,  # Deterministic generation
                    temperature=None,  # Explicitly set to None to override defaults
                    top_p=None,       # Explicitly set to None to override defaults
                )
            finally:
                # Stop the spinner when inference is done (even if there's an error)
                stop_spinner.set()
                spinner_thread.join(timeout=1.0)  # Wait for spinner to finish
            
            # Based on the example in llama.py, we extract the assistant's response
            # which is the last message in the generated output
            response_content = ""
            try:
                # Get the last message from the generated conversation
                last_message = outputs[0]["generated_text"][-1]
                
                # The last message should be a dict with the assistant's response
                if isinstance(last_message, dict) and "content" in last_message:
                    response_content = last_message["content"].strip()
                else:
                    # If it's not a dict with content, try to use it directly
                    response_content = str(last_message).strip()
            except (IndexError, KeyError, AttributeError) as e:
                print(f"Error extracting response from model output: {e}")
                print(f"Output format: {type(outputs)}")
                # Try a fallback approach
                if isinstance(outputs, list) and len(outputs) > 0:
                    print("Attempting fallback extraction...")
                    response_content = str(outputs[0])
            
            # Extract JSON from the response
            start_idx = response_content.find('[')
            end_idx = response_content.rfind(']') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_content[start_idx:end_idx]
                try:
                    relationships = json.loads(json_str)
                    
                    # Ensure confidence is a float
                    for rel in relationships:
                        if 'confidence' in rel:
                            try:
                                rel['confidence'] = float(rel['confidence'])
                            except (ValueError, TypeError):
                                rel['confidence'] = 0.5
                        else:
                            rel['confidence'] = 1.0
                            
                    return relationships
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON: {e}")
                    print(f"JSON string: {json_str[:100]}...")
                    return []
            else:
                print(f"Error: Could not find JSON array in model response")
                print(f"Response content: {response_content[:100]}...")
                return []
            
        except Exception as e:
            print(f"Error during model inference or processing: {e}")
            return [] 