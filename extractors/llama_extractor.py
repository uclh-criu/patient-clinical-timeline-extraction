import os
import json
import torch
from extractors.base_extractor import BaseRelationExtractor
from tqdm import tqdm

class LlamaExtractor(BaseRelationExtractor):
    """
    Relation extractor that uses Llama 3.2 3B model locally to identify 
    entity-date relationships in clinical notes.
    """
    
    def __init__(self, config):
        """
        Initialize the Llama extractor.
        
        Args:
            config: Configuration object or dict with necessary parameters.
        """
        super().__init__(config)  # Call the parent constructor
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
    
    def extract(self, text, entities=None, note_id=None, patient_id=None):
        """
        Extract relationships using the Llama 3.2 model.
        
        Args:
            text (str): The clinical note text.
            entities (tuple, optional): A tuple of (entities_list, dates) if already extracted.
            note_id (int, optional): The ID of the note being processed.
            patient_id (str, optional): The ID of the patient the note belongs to.
            
        Returns:
            list: A list of dictionaries, each representing a relationship:
                {
                    'entity_label': str,     # The entity text
                    'entity_category': str,  # The entity category
                    'date': str,             # The date text
                    'confidence': float      # Model prediction confidence
                }
        """
        import time
        import hashlib
        
        # Use provided IDs if available, otherwise generate a hash
        if note_id is not None and patient_id is not None:
            row_id = f"Patient {patient_id}, Note {note_id}"
        else:
            # Generate a unique ID for this text to help identify problematic rows
            text_hash = hashlib.md5(text[:100].encode()).hexdigest()[:8]
            row_id = f"Hash {text_hash}"
        
        print(f"\nProcessing {row_id} (length: {len(text)} chars, first 50: '{text[:50].replace(chr(10), ' ')}...')")
        
        start_time = time.time()
        if self.pipeline is None:
            print("Llama model not initialized. Call load() first.")
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
                entities_info.append({
                    "entity_label": entity.get('label', ''),
                    "entity_category": entity.get('category', 'unknown'),
                    "position": entity.get('start', 0)
                })
            else:
                # Handle tuple formats: (label, position) or (label, position, category)
                if len(entity) == 3:
                    # Multi-entity format: (label, position, category)
                    label, position, category = entity
                    entities_info.append({
                        "entity_label": label,
                        "entity_category": category,
                        "position": position
                    })
                elif len(entity) == 2:
                    # Legacy format: (label, position)
                    label, position = entity
                    entities_info.append({
                        "entity_label": label,
                        "entity_category": "disorder",  # Default category
                        "position": position
                    })
        
        # Extract parsed date, raw date string, and position
        dates_info = [{"parsed_date": d[0], "raw_date": d[1], "position": d[2]} for d in dates]
        
        # Construct the messages for the model
        system_prompt = "You are a medical AI assistant specialized in analyzing clinical notes to find relationships between medical entities and dates."
        
        user_prompt = f"""
        You are a medical AI assistant. Your task is to find relationships between medical entities and dates in a clinical note.
        
        For each entity in 'Entities Info', find the most relevant date from 'Dates Info'.

        **CRITICAL INSTRUCTIONS:**
        1.  **Return ONLY a JSON array.** Do not include any other text, explanations, or formatting.
        2.  **ONLY include entities with a valid, non-null date.** If you cannot find a date for an entity, you MUST OMIT it from your response entirely.
        3.  The `date` field in the JSON MUST be a string in "YYYY-MM-DD" format. It cannot be `null`.

        **EXAMPLE OF GOOD OUTPUT:**
        ```json
        [
          {{
            "entity_label": "hematoma",
            "entity_category": "disorder",
            "date": "2019-04-18",
            "confidence": 0.9
          }}
        ]
        ```

        **EXAMPLE OF BAD OUTPUT (DO NOT DO THIS):**
        ```json
        [
          {{
            "entity_label": "headache",
            "entity_category": "symptom",
            "date": null,
            "confidence": 0.1
          }}
        ]
        ```

        ---
        Clinical Note: {text}
        ---
        Entities Info: {entities_info}
        ---
        Dates Info: {dates_info}
        ---

        Your JSON response:
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            import sys
            import time
            import threading
            from queue import Queue, Empty

            # Timeout in seconds (10 minutes)
            timeout_seconds = 600

            # Queue to hold the result from the model thread
            result_queue = Queue()

            # This function will run in a separate thread
            def run_model_in_thread():
                try:
                    outputs = self.pipeline(
                        messages,
                        max_new_tokens=2000,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                    )
                    result_queue.put(outputs)
                except Exception as e:
                    result_queue.put(e)

            # Start the model inference in a separate thread
            model_thread = threading.Thread(target=run_model_in_thread)
            model_thread.daemon = True
            model_thread.start()

            # --- Spinner setup ---
            stop_spinner = threading.Event()
            def spinner_function():
                chars = "|/-\\"
                i = 0
                spinner_start_time = time.time()
                while not stop_spinner.is_set() and model_thread.is_alive():
                    elapsed = int(time.time() - spinner_start_time)
                    mins, secs = divmod(elapsed, 60)
                    timeformat = f"{mins:02d}:{secs:02d}"
                    sys.stdout.write(f"\rGenerating response... {chars[i % len(chars)]} [{timeformat} elapsed]")
                    sys.stdout.flush()
                    time.sleep(0.1)
                    i += 1
            
            spinner_thread = threading.Thread(target=spinner_function)
            spinner_thread.daemon = True
            spinner_thread.start()
            # --- End of spinner setup ---
            
            outputs = None
            try:
                # Wait for the result from the queue, with a timeout
                outputs = result_queue.get(timeout=timeout_seconds)

                if isinstance(outputs, Exception):
                    raise outputs
            
            except Empty:
                # This block executes if result_queue.get() times out
                print(f"\nERROR: Model inference for row with ID {row_id} timed out after {timeout_seconds / 60:.0f} minutes. Skipping.")
                print(f"Warning: The timed-out model process might still be running in the background.", file=sys.stderr)
                return []

            finally:
                # Stop the spinner thread
                stop_spinner.set()
                spinner_thread.join(timeout=1.0)
                if outputs is not None:
                    sys.stdout.write("\rInference complete!                                  \n")
                    sys.stdout.flush()
            
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
            
            # --- DEBUG: Print the raw model response for analysis ---
            print(f"\n--- Model Response for Note ID {row_id} ---")
            print(response_content[:500] + "..." if len(response_content) > 500 else response_content)
            print("--- End Model Response ---\n")
            
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
                    
                    # --- DEBUG: Print each prediction as it's processed ---
                    print(f"\n=== Predictions for Note ID {row_id} ===")
                    if relationships:
                        for i, rel in enumerate(relationships):
                            print(f"  Prediction {i+1}: {rel}")
                    else:
                        print("  No predictions found in model response")
                    print(f"=== End Predictions for Note ID {row_id} ===\n")
                    
                    # Filter out any relationships with None dates
                    valid_relationships = [rel for rel in relationships if rel.get('date') is not None]
                    if len(valid_relationships) < len(relationships):
                        print(f"  Filtered out {len(relationships) - len(valid_relationships)} relationships with None dates")
                            
                    return valid_relationships
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