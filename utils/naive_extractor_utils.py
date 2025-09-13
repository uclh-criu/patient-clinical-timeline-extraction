def naive_extraction(entities_list, dates, max_distance):
    """
    Simple proximity-based relationship extraction. Finds nearest date for each entity within max_distance.
    Note: only ever assigns an entity one date (no more or less) as long as there is one within max distance.
    Args: entity and date dicts with character positions.
    Returns: list of relationship dicts showing linked entities and dates, as well as distance.  
    """
    relationships = []
    
    for entity in entities_list:
        entity_pos = entity['start']
        
        # Find closest date
        closest_date = None
        min_distance = float('inf')
        
        for date_info in dates:
            date_pos = date_info['start']
            distance = abs(entity_pos - date_pos)
            
            if distance < min_distance and distance <= max_distance:
                min_distance = distance
                closest_date = date_info['value']
                
        # Add relationship if found
        if closest_date:
            relationships.append({
                'entity_label': entity['value'],
                'date': closest_date,
                'distance': min_distance
            })
    
    return relationships