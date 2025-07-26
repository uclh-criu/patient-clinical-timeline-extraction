import os
import json
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv
from utils.inference_utils import safe_json_loads

def extract_relative_dates_llm(text, document_timestamp, config):
    """
    Extract relative date references from text using an LLM.
    Returns a list of tuples: (parsed_date_str, raw_phrase_str, start_position)
    compatible with the existing dates format.
    
    Args:
        text (str): The clinical note text
        document_timestamp (datetime): The timestamp of the document for reference
        config: Configuration object with LLM settings
        
    Returns:
        list: A list of date tuples (parsed_date_str, raw_phrase_str, start_position)
    """
    # Check if relative date extraction is enabled
    if not hasattr(config, 'ENABLE_RELATIVE_DATE_EXTRACTION') or not config.ENABLE_RELATIVE_DATE_EXTRACTION:
        return []
        
    # Check inputs
    if not text or pd.isna(text) or not document_timestamp:
        return []
    
    # Truncate text if longer than the configured context window
    max_context = getattr(config, 'RELATIVE_DATE_CONTEXT_WINDOW', 1000)
    if len(text) > max_context:
        text = text[:max_context]
    
    # Determine which LLM to use
    llm_model = getattr(config, 'RELATIVE_DATE_LLM_MODEL', 'llama')
    
    if llm_model.lower() == 'openai':
        return extract_relative_dates_openai(text, document_timestamp, config)
    elif llm_model.lower() == 'llama':
        return extract_relative_dates_llama(text, document_timestamp, config)
    else:
        print(f"Warning: Unknown RELATIVE_DATE_LLM_MODEL: {llm_model}")
        return []

def extract_relative_dates_openai(text, document_timestamp, config):
    """
    Extract relative dates using OpenAI API.
    
    Args:
        text (str): The clinical note text
        document_timestamp (datetime): The timestamp of the document for reference
        config: Configuration object with OpenAI settings
        
    Returns:
        list: A list of date tuples (parsed_date_str, raw_phrase_str, start_position)
    """
    try:
        # Load environment variables for API key
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        
        # Reduce verbosity
        debug_mode = getattr(config, 'DEBUG_MODE', False)
        if debug_mode:
            print("Attempting to use OpenAI API for relative date extraction...")
        
        if not api_key:
            print("Error: OPENAI_API_KEY not found in .env file or environment variables.")
            return []
        
        if api_key == "your_actual_api_key_here" or api_key == "your_api_key_here":
            print("Error: You need to replace the placeholder in .env with your actual OpenAI API key.")
            return []
            
        if debug_mode:
            print(f"API key found (starts with: {api_key[:4]}{'*' * 20})")
        
        # Import OpenAI
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            if debug_mode:
                print("OpenAI client initialized successfully")
        except ImportError:
            print("Error: openai package not installed. Install with 'pip install openai'.")
            return []
        
        # Get model name from config or use default
        model_name = getattr(config, 'RELATIVE_DATE_OPENAI_MODEL', 'gpt-3.5-turbo')
        if debug_mode:
            print(f"Using OpenAI model: {model_name}")
        
        # Format the timestamp for the prompt
        timestamp_str = document_timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        # Construct the prompt
        prompt = f"""
You are a medical AI assistant specialized in extracting temporal expressions from clinical notes.

TASK: Find ALL relative date expressions in the clinical note below and convert them to absolute dates.

DOCUMENT CREATION DATE: {timestamp_str}

CLINICAL NOTE TEXT:
"{text}"

WHAT TO FIND (Relative Date Expressions):
- Time ago: "6 months ago", "3 years ago", "last week", "yesterday", "three days prior"
- Duration references: "3-month history", "over the past 2 weeks", "for the past year", "2-week history"
- Contextual time: "last Tuesday", "next month", "this past year", "earlier this week"  
- Prior/before references: "5 years prior", "diagnosed 2 years before", "2 weeks prior to presentation"
- Future references: "in 6 months", "follow up in 3 weeks", "scheduled next month", "in 1 week"
- Vague temporal: "recently", "last visit", "previous consultation", "last month"

WHAT NOT TO FIND (Absolute Dates - IGNORE THESE):
- Specific dates: "2023-08-15", "15/11/2023", "October 3, 2023", "22 Aug 2023"
- Birth dates: "DOB: 1975-03-15"
- Any date in YYYY-MM-DD, DD/MM/YYYY, MM/DD/YYYY, or "Month DD, YYYY" format

INSTRUCTIONS:
1. Search the ENTIRE text for phrases that express time RELATIVE to the document date
2. IGNORE any absolute/specific dates - only find expressions that are relative to "now"
3. For each relative phrase found, note its EXACT start position in the original text
4. Calculate the absolute date based on the document creation date
5. Be thorough - include duration patterns like "3-month history", "over the past X", "in X weeks/months"

RETURN FORMAT: JSON array with objects containing:
- "phrase": exact text of the relative date expression
- "start_index": character position where phrase starts in the text
- "calculated_date": absolute date in YYYY-MM-DD format

COMPREHENSIVE EXAMPLES:
Text: "Patient has 3-month history of symptoms. Surgery was 2 years ago. Follow-up in 6 months. Over the past 2 weeks, symptoms worsened. Last visit was on 2023-06-01."
Document date: 2023-06-09
Result:
[
  {{"phrase": "3-month history", "start_index": 12, "calculated_date": "2023-03-09"}},
  {{"phrase": "2 years ago", "start_index": 48, "calculated_date": "2021-06-09"}},
  {{"phrase": "in 6 months", "start_index": 75, "calculated_date": "2023-12-09"}},
  {{"phrase": "Over the past 2 weeks", "start_index": 90, "calculated_date": "2023-05-26"}},
  {{"phrase": "Last visit", "start_index": 124, "calculated_date": "2023-06-01"}}
]
Note: "2023-06-01" would NOT be included as it's an absolute date.

If NO relative dates found, return: []

JSON RESULT:"""
        
        if debug_mode:
            print("Sending request to OpenAI API...")
        
        # Call the OpenAI API
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a medical AI assistant specialized in extracting temporal expressions from clinical notes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=1000
            )
            if debug_mode:
                print("Received response from OpenAI API")
        except Exception as api_error:
            print(f"OpenAI API error: {api_error}")
            return []
        
        # Extract the response text
        response_text = response.choices[0].message.content.strip()
        if debug_mode:
            print(f"Response text length: {len(response_text)} characters")
            print(f"Full OpenAI response text:")
            print("=" * 50)
            print(response_text)
            print("=" * 50)
        
        # Find the JSON array in the response
        start_idx = response_text.find('[')
        end_idx = response_text.rfind(']') + 1
        
        if start_idx >= 0 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            if debug_mode:
                print(f"Found JSON array: {json_str[:100]}...")
            
            try:
                dates_data = safe_json_loads(json_str)
                if dates_data and debug_mode:
                    print(f"Successfully parsed JSON with {len(dates_data)} results")
                
                # Convert to the expected tuple format
                relative_dates = []
                for item in dates_data:
                    phrase = item.get('phrase', '')
                    start_index = item.get('start_index', 0)
                    calculated_date = item.get('calculated_date', '')
                    
                    # Only add valid entries
                    if phrase and calculated_date:
                        # Filter out phrases that look like absolute dates
                        import re
                        absolute_date_patterns = [
                            r'\b\d{4}-\d{1,2}-\d{1,2}\b',          # YYYY-MM-DD
                            r'\b\d{1,2}/\d{1,2}/\d{4}\b',          # DD/MM/YYYY or MM/DD/YYYY
                            r'\b\w+ \d{1,2}, \d{4}\b',             # Month DD, YYYY
                            r'\b\d{1,2} \w+ \d{4}\b',              # DD Month YYYY
                        ]
                        
                        is_absolute_date = any(re.search(pattern, phrase) for pattern in absolute_date_patterns)
                        
                        if not is_absolute_date:
                            relative_dates.append((calculated_date, phrase, start_index))
                
                return relative_dates
            except json.JSONDecodeError as e:
                print(f"Error parsing OpenAI JSON response: {e}")
                return []
        else:
            if debug_mode:
                print("No JSON array found in OpenAI response")
                if len(response_text) < 200:
                    print(f"Full response was: {response_text}")
            return []
            
    except Exception as e:
        print(f"Error in OpenAI relative date extraction: {e}")
        if getattr(config, 'DEBUG_MODE', False):
            import traceback
            traceback.print_exc()
        return []

def extract_relative_dates_llama(text, document_timestamp, config):
    """
    Extract relative dates using Llama model.
    
    Args:
        text (str): The clinical note text
        document_timestamp (datetime): The timestamp of the document for reference
        config: Configuration object with Llama settings
        
    Returns:
        list: A list of date tuples (parsed_date_str, raw_phrase_str, start_position)
    """
    try:
        # Check if transformers package is installed
        try:
            from transformers import pipeline
            import torch
        except ImportError:
            print("Error: transformers package not installed. Install with 'pip install transformers'.")
            return []
        
        # Get model path from config
        model_path = getattr(config, 'LLAMA_MODEL_PATH', './Llama-3.2-3B-Instruct')
        
        # Initialize the pipeline
        try:
            # Create a loading indicator since model loading can take time
            with tqdm(total=100, desc="Loading Llama model", unit="%") as pbar:
                print(f"Loading Llama model for relative date extraction from {model_path}...")
                
                # Update progress to 25% - started loading
                pbar.update(25)
                
                # Load the model
                pipe = pipeline(
                    "text-generation",
                    model=model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                
                # Update progress to 100% - done loading
                pbar.update(75)
        except Exception as e:
            print(f"Error loading Llama model: {e}")
            return []
        
        # Format the timestamp for the prompt
        timestamp_str = document_timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        # Construct the messages for the model
        system_prompt = "You are a medical AI assistant specialized in extracting temporal expressions from clinical notes."
        
        user_prompt = f"""
TASK: Find ALL relative date expressions in the clinical note below and convert them to absolute dates.

DOCUMENT CREATION DATE: {timestamp_str}

CLINICAL NOTE TEXT:
"{text}"

WHAT TO FIND (Relative Date Expressions):
- Time ago: "6 months ago", "3 years ago", "last week", "yesterday", "three days prior"
- Duration references: "3-month history", "over the past 2 weeks", "for the past year", "2-week history"
- Contextual time: "last Tuesday", "next month", "this past year", "earlier this week"  
- Prior/before references: "5 years prior", "diagnosed 2 years before", "2 weeks prior to presentation"
- Future references: "in 6 months", "follow up in 3 weeks", "scheduled next month", "in 1 week"
- Vague temporal: "recently", "last visit", "previous consultation", "last month"

WHAT NOT TO FIND (Absolute Dates - IGNORE THESE):
- Specific dates: "2023-08-15", "15/11/2023", "October 3, 2023", "22 Aug 2023"
- Birth dates: "DOB: 1975-03-15"
- Any date in YYYY-MM-DD, DD/MM/YYYY, MM/DD/YYYY, or "Month DD, YYYY" format

INSTRUCTIONS:
1. Search the ENTIRE text for phrases that express time RELATIVE to the document date
2. IGNORE any absolute/specific dates - only find expressions that are relative to "now"
3. For each relative phrase found, note its EXACT start position in the original text
4. Calculate the absolute date based on the document creation date
5. Be thorough - include duration patterns like "3-month history", "over the past X", "in X weeks/months"

RETURN FORMAT: JSON array with objects containing:
- "phrase": exact text of the relative date expression
- "start_index": character position where phrase starts in the text
- "calculated_date": absolute date in YYYY-MM-DD format

COMPREHENSIVE EXAMPLES:
Text: "Patient has 3-month history of symptoms. Surgery was 2 years ago. Follow-up in 6 months. Over the past 2 weeks, symptoms worsened. Last visit was on 2023-06-01."
Document date: 2023-06-09
Result:
[
  {{"phrase": "3-month history", "start_index": 12, "calculated_date": "2023-03-09"}},
  {{"phrase": "2 years ago", "start_index": 48, "calculated_date": "2021-06-09"}},
  {{"phrase": "in 6 months", "start_index": 75, "calculated_date": "2023-12-09"}},
  {{"phrase": "Over the past 2 weeks", "start_index": 90, "calculated_date": "2023-05-26"}},
  {{"phrase": "Last visit", "start_index": 124, "calculated_date": "2023-06-01"}}
]
Note: "2023-06-01" would NOT be included as it's an absolute date.

If NO relative dates found, return: []

JSON RESULT:"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Run inference with the model
        outputs = pipe(
            messages,
            max_new_tokens=1000,
            do_sample=False,
            temperature=None,
        )
        
        # Extract response content
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
            print(f"Error extracting response from Llama model output: {e}")
            return []
        
        # Extract JSON from the response
        start_idx = response_content.find('[')
        end_idx = response_content.rfind(']') + 1
        
        if start_idx >= 0 and end_idx > start_idx:
            json_str = response_content[start_idx:end_idx]
            
            try:
                dates_data = safe_json_loads(json_str)
                
                # Convert to the expected tuple format
                relative_dates = []
                for item in dates_data:
                    phrase = item.get('phrase', '')
                    start_index = item.get('start_index', 0)
                    calculated_date = item.get('calculated_date', '')
                    
                    # Only add valid entries
                    if phrase and calculated_date:
                        # Filter out phrases that look like absolute dates
                        import re
                        absolute_date_patterns = [
                            r'\b\d{4}-\d{1,2}-\d{1,2}\b',          # YYYY-MM-DD
                            r'\b\d{1,2}/\d{1,2}/\d{4}\b',          # DD/MM/YYYY or MM/DD/YYYY
                            r'\b\w+ \d{1,2}, \d{4}\b',             # Month DD, YYYY
                            r'\b\d{1,2} \w+ \d{4}\b',              # DD Month YYYY
                        ]
                        
                        is_absolute_date = any(re.search(pattern, phrase) for pattern in absolute_date_patterns)
                        
                        if not is_absolute_date:
                            relative_dates.append((calculated_date, phrase, start_index))
                
                return relative_dates
            except json.JSONDecodeError as e:
                print(f"Error parsing Llama JSON response: {e}")
                return []
        else:
            print("No JSON array found in Llama response")
            return []
    
    except Exception as e:
        print(f"Error in Llama relative date extraction: {e}")
        return []