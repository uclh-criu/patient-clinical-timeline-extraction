def naive_extraction(note_text, entities_list, dates, max_distance=200):
    """Simple proximity-based relationship extraction"""
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
                closest_date = date_info['parsed']
                
        # Add relationship if found
        if closest_date:
            relationships.append({
                'entity_label': entity['label'],
                'entity_category': entity.get('category', 'disorder'),
                'date': closest_date,
                'confidence': 1.0,
                'distance': min_distance
            })
    
    return relationships
