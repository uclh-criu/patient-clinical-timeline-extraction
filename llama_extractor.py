def llama_extraction(text, generator):
    """Predict relationship using Llama model"""
    
    # Generate response
    outputs = generator(text, max_new_tokens=10, do_sample=False)
    full_response = outputs[0]['generated_text']
    
    # Extract only the newly generated part (remove original prompt)
    #new_text = full_response[len(text):].strip()
    
    return full_response

def process_llama_response(response, confidence_yes=0.8, confidence_no=0.8, confidence_unknown=0.5):
    """Convert Llama text response to prediction and confidence"""
    response_upper = response.upper()
    
    if "YES" in response_upper and "NO" not in response_upper.split("YES")[0]:
        return 1, confidence_yes  # Relationship exists
    elif "NO" in response_upper:
        return 0, confidence_no   # No relationship
    else:
        return 0, confidence_unknown  # Unclear, default to no relationship