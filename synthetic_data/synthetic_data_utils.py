import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import re


def seed_all(seed: int) -> None:
    """
    Seed python's random and numpy RNGs for reproducibility.
    Args:
        seed (int): Random seed value.
    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)


def sample_note_length(length_stats: Dict[str, float]) -> int:
    """
    Sample a note length from real data statistics (mean, std).
    Args:
        length_stats (dict): Dict with 'mean' and 'std' for note length.
    Returns:
        int: Sampled note length (at least 1).
    """
    return max(1, int(np.random.normal(length_stats['mean'], length_stats['std'])))


def sample_count(stats: Dict[str, float]) -> int:
    """
    Sample a count (e.g., entities or dates per note) from real data statistics.
    Args:
        stats (dict): Dict with 'mean' and 'std'.
    Returns:
        int: Sampled count (at least 1).
    """
    return max(1, int(np.random.normal(stats['mean'], stats['std'])))


def sample_entities(entity_freqs: Dict[str, int], n_entities: int) -> List[str]:
    """
    Sample entities according to their real frequency distribution.
    Args:
        entity_freqs (dict): {entity_label: count}
        n_entities (int): Number of entities to sample.
    Returns:
        List[str]: List of sampled entity labels.
    """
    entities = list(entity_freqs.keys())
    weights = np.array([entity_freqs[e] for e in entities], dtype=float)
    weights /= weights.sum() if weights.sum() > 0 else 1.0
    return random.choices(entities, weights=weights, k=n_entities)


def generate_base_note(target_length: int, filler_phrases: List[str]) -> str:
    """
    Generate a base note of approximately the target length using random filler phrases.
    Args:
        target_length (int): Desired note length in characters.
        filler_phrases (List[str]): List of clinical phrases to sample from.
    Returns:
        str: Generated note text.
    """
    note = ""
    while len(note) < target_length:
        note += random.choice(filler_phrases) + " "
    return note[:target_length]


def insert_at_word_boundaries(note: str, text: str, min_distance: int, used_positions: List[int]) -> Optional[int]:
    """
    Choose an insertion index aligned to a word/sentence boundary and respecting min distance to used positions.
    Args:
        note (str): Note text.
        text (str): Text to insert (used for length).
        min_distance (int): Minimum distance from other insertions.
        used_positions (List[int]): Existing insertion anchor starts.
    Returns:
        Optional[int]: Chosen insertion index or None if not found.
    """
    boundaries = []
    for i, ch in enumerate(note):
        if ch in [' ', '\n', '.', '!', '?'] and i + len(text) < len(note):
            boundaries.append(i)
    random.shuffle(boundaries)
    for pos in boundaries:
        if all(abs(pos - u) > min_distance for u in used_positions):
            return pos
    return None


def build_entity_sentence(entity: str, templates: Optional[List[str]] = None) -> str:
    """
    Build a sentence that mentions an entity using simple templates.
    Args:
        entity (str): Entity label.
        templates (List[str], optional): Sentence templates with {entity} placeholder.
    Returns:
        str: Generated sentence containing the entity.
    """
    templates = templates or [
        "Assessment includes {entity}.",
        "{entity} was discussed.",
        "Patient has a history of {entity}.",
        "Noted {entity} in medical history."
    ]
    return random.choice(templates).format(entity=entity)


def build_entity_date_sentence(entity: str, date_str: str, templates: Optional[List[str]] = None) -> str:
    """
    Build a sentence that co-mentions an entity and a date using templates.
    Args:
        entity (str): Entity label.
        date_str (str): Date string as appears in note (e.g., "(01 Jan 2024)").
    Returns:
        str: Generated sentence containing entity and date.
    """
    templates = templates or [
        "Diagnosis of {entity} documented on {date}.",
        "{entity} noted on {date}.",
        "On {date}, {entity} was recorded.",
        "{entity} confirmed on {date}."
    ]
    return random.choice(templates).format(entity=entity, date=date_str)


def find_insertion_points(note: str, min_distance: int, item_length: int) -> List[int]:
    """
    Find valid insertion points in the note for non-overlapping placement (grid-based).
    Args:
        note (str): The note text.
        min_distance (int): Minimum distance between insertions.
        item_length (int): Length of the item to insert.
    Returns:
        List[int]: List of valid insertion indices.
    """
    points = [i for i in range(0, len(note) - item_length, min_distance)]
    return points


def insert_entities_nonoverlapping(note: str, entities: List[str], min_distance: int = 30) -> List[Dict[str, Any]]:
    """
    Insert entities at non-overlapping positions in the note (grid-based, may split words).
    Args:
        note (str): The note text.
        entities (List[str]): List of entity labels.
        min_distance (int): Minimum distance between entities.
    Returns:
        List[Dict]: List of entity dicts with 'label', 'start', 'end'.
    """
    used = []
    entity_spans = []
    for entity in entities:
        item_length = len(entity)
        candidates = find_insertion_points(note, min_distance, item_length)
        candidates = [c for c in candidates if all(abs(c - u) > min_distance for u in used)]
        if not candidates:
            continue
        pos = random.choice(candidates)
        entity_spans.append({'label': entity, 'start': pos, 'end': pos + item_length})
        used.append(pos)
    return entity_spans


def insert_entities_at_boundaries(note: str, entities: List[str], min_distance: int = 30) -> List[Dict[str, Any]]:
    """
    Insert entities at word/sentence boundaries ensuring non-overlap.
    Args:
        note (str): The note text.
        entities (List[str]): List of entity labels.
        min_distance (int): Minimum distance between entities.
    Returns:
        List[Dict]: List of entity dicts with 'label', 'start', 'end'.
    """
    used = []
    entity_spans = []
    for entity in entities:
        pos = insert_at_word_boundaries(note, entity, min_distance, used)
        if pos is None:
            continue
        entity_spans.append({'label': entity, 'start': pos, 'end': pos + len(entity)})
        used.append(pos)
    return entity_spans


def insert_dates_nonoverlapping(note: str, n_dates: int, min_distance: int = 30, date_format: str = "(01 Jan 2020)") -> List[Dict[str, Any]]:
    """
    Insert date mentions at non-overlapping positions in the note (grid-based).
    Args:
        note (str): The note text.
        n_dates (int): Number of dates to insert.
        min_distance (int): Minimum distance between dates.
        date_format (str): Format string for dates (placeholder).
    Returns:
        List[Dict]: List of date dicts with 'original', 'parsed', 'start', 'end'.
    """
    used = []
    date_spans = []
    for i in range(n_dates):
        date_str = f"(01 Jan 202{i})"
        item_length = len(date_str)
        candidates = find_insertion_points(note, min_distance, item_length)
        candidates = [c for c in candidates if all(abs(c - u) > min_distance for u in used)]
        if not candidates:
            continue
        pos = random.choice(candidates)
        date_spans.append({'original': date_str, 'parsed': f"202{i}-01-01", 'start': pos, 'end': pos + item_length})
        used.append(pos)
    return date_spans


def insert_dates_at_boundaries(note: str, n_dates: int, min_distance: int = 30, formats: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Insert date mentions at word/sentence boundaries with simple format variants.
    Args:
        note (str): The note text.
        n_dates (int): Number of dates to insert.
        min_distance (int): Minimum distance between dates.
        formats (List[str], optional): List of date format strings to sample (textual placeholders).
    Returns:
        List[Dict]: List of date dicts with 'original', 'parsed', 'start', 'end'.
    """
    formats = formats or [
        "(01 Jan 202{y})",
        "(202{y}-01-01)",
        "(01/01/202{y})",
        "(01.01.202{y})"
    ]
    used = []
    date_spans = []
    for i in range(n_dates):
        fmt = random.choice(formats)
        date_str = fmt.format(y=i)
        pos = insert_at_word_boundaries(note, date_str, min_distance, used)
        if pos is None:
            continue
        date_spans.append({'original': date_str, 'parsed': f"202{i}-01-01", 'start': pos, 'end': pos + len(date_str)})
        used.append(pos)
    return date_spans


def link_relationships_by_proximity(entities: List[Dict[str, Any]], dates: List[Dict[str, Any]], max_distance: int = 500) -> List[Dict[str, Any]]:
    """
    Create positive relationships by linking each entity to the nearest date within a max distance.
    Args:
        entities (List[Dict]): Entity spans.
        dates (List[Dict]): Date spans.
        max_distance (int): Maximum allowed distance to consider a positive pair.
    Returns:
        List[Dict]: Relationship dicts similar to gold pairs.
    """
    rels = []
    for e in entities:
        if not dates:
            continue
        d = min(dates, key=lambda x: abs(e['start'] - x['start']))
        if abs(e['start'] - d['start']) <= max_distance:
            rels.append({'date': d['parsed'], 'date_position': d['start'], 'diagnoses': [{'diagnosis': e['label'], 'position': e['start']} ]})
    return rels


def generate_negative_pairs(entities: List[Dict[str, Any]], dates: List[Dict[str, Any]], positive_rels: List[Dict[str, Any]], ratio: float = 1.0) -> List[Tuple[str, str]]:
    """
    Generate negative (entity, date) pairs not present in positives.
    Args:
        entities (List[Dict]): Entity spans.
        dates (List[Dict]): Date spans.
        positive_rels (List[Dict]): Positive relationships.
        ratio (float): Negatives per positive target ratio.
    Returns:
        List[Tuple[str, str]]: List of (entity_label, date_parsed) negatives.
    """
    pos_set = set((d['diagnosis'], r['date']) for r in positive_rels for d in r['diagnoses'])
    all_pairs = [(e['label'], d['parsed']) for e in entities for d in dates]
    negs = [p for p in all_pairs if p not in pos_set]
    k = int(len(positive_rels) * ratio)
    random.shuffle(negs)
    return negs[:k]


def ensure_min_pair_distance(note: str, entity_span: Dict[str, Any], date_span: Dict[str, Any], min_distance: int, filler_phrases: List[str]) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """
    Ensure a minimum character distance between an entity and a date by inserting filler text between them.
    Args:
        note (str): Original note text.
        entity_span (dict): {'label','start','end'} for entity.
        date_span (dict): {'original','parsed','start','end'} for date.
        min_distance (int): Minimum required distance.
        filler_phrases (List[str]): Sentences to insert as filler.
    Returns:
        Tuple[str, dict, dict]: (updated_note, updated_entity_span, updated_date_span).
    """
    e_end = entity_span['end']
    d_start = date_span['start']
    gap = abs(d_start - e_end)
    if gap >= min_distance:
        return note, entity_span, date_span
    insert_pos = min(e_end, d_start)
    filler = " " + random.choice(filler_phrases) + " "
    new_note = note[:insert_pos] + filler + note[insert_pos:]
    delta = len(filler)
    def shift_span(span: Dict[str, Any]) -> Dict[str, Any]:
        s = dict(span)
        if s['start'] >= insert_pos:
            s['start'] += delta
            s['end'] += delta
        return s
    entity_span_u = shift_span(entity_span)
    date_span_u = shift_span(date_span)
    return new_note, entity_span_u, date_span_u


def match_date_density(dates: List[Dict[str, Any]], relationships: List[Dict[str, Any]], target_avg: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Reduce date mentions and their relationships to approach a target average.
    Args:
        dates (List[Dict]): Date spans.
        relationships (List[Dict]): Positive relationships.
        target_avg (float): Desired average dates per note.
    Returns:
        Tuple[List[Dict], List[Dict]]: (filtered_dates, filtered_relationships)
    """
    keep_n = int(target_avg)
    if keep_n >= len(dates):
        return dates, relationships
    dates_sorted = sorted(dates, key=lambda x: x['start'])
    kept = set(d['parsed'] for d in dates_sorted[:keep_n])
    dates_out = [d for d in dates if d['parsed'] in kept]
    rels_out = [r for r in relationships if r['date'] in kept]
    return dates_out, rels_out


def insert_random_filler(note: str, filler_phrases: List[str]) -> Tuple[str, int, int]:
    """
    Insert a single filler sentence at a sentence/line boundary.
    Args:
        note (str): Original note text.
        filler_phrases (List[str]): Pool of filler sentences.
    Returns:
        Tuple[str, int, int]: (new_note, insert_position, insert_length)
    """
    filler = " " + random.choice(filler_phrases)
    sentence_ends = [m.end() for m in re.finditer(r'\.(?=\s|$)', note)]
    line_ends = [m.end() for m in re.finditer(r'\n', note)]
    possible_positions = sentence_ends + line_ends
    insert_position = random.choice(possible_positions) if possible_positions else random.randint(0, len(note))
    new_text = note[:insert_position] + filler + note[insert_position:]
    return new_text, insert_position, len(filler)


def shift_spans_after_insertion(spans: List[Dict[str, Any]], insert_pos: int, insert_len: int) -> List[Dict[str, Any]]:
    """
    Shift spans (with 'start' and 'end') occurring at/after insertion position by insert_len.
    Args:
        spans (List[Dict]): List of spans with 'start' and 'end'.
        insert_pos (int): Insertion index.
        insert_len (int): Inserted text length.
    Returns:
        List[Dict]: Updated spans.
    """
    out = []
    for s in spans:
        s2 = dict(s)
        if s2['start'] >= insert_pos:
            s2['start'] += insert_len
            s2['end'] += insert_len
        out.append(s2)
    return out


def update_relationship_positions(relationships: List[Dict[str, Any]], entities: List[Dict[str, Any]], dates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Update relationship positions based on current entity/date spans.
    Args:
        relationships (List[Dict]): Relationship dicts with 'date' and 'diagnoses'.
        entities (List[Dict]): Current entity spans.
        dates (List[Dict]): Current date spans.
    Returns:
        List[Dict]: Relationships with refreshed positions.
    """
    entity_pos = {e['label']: e['start'] for e in entities}
    date_pos = {d['parsed']: d['start'] for d in dates}
    out = []
    for r in relationships:
        r2 = dict(r)
        if r2['date'] in date_pos:
            r2['date_position'] = date_pos[r2['date']]
        diags = []
        for d in r2.get('diagnoses', []):
            d2 = dict(d)
            if d2['diagnosis'] in entity_pos:
                d2['position'] = entity_pos[d2['diagnosis']]
            diags.append(d2)
        r2['diagnoses'] = diags
        out.append(r2)
    return out


def ensure_min_distance_for_all_pairs(note: str, entities: List[Dict[str, Any]], dates: List[Dict[str, Any]], relationships: List[Dict[str, Any]], min_distance: int, filler_phrases: List[str]) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Enforce a minimum distance for all positive entity-date pairs by inserting filler between them.
    Args:
        note (str): Original note text.
        entities (List[Dict]): Entity spans.
        dates (List[Dict]): Date spans.
        relationships (List[Dict]): Positive relationships.
        min_distance (int): Minimum distance.
        filler_phrases (List[str]): Filler sentences to insert.
    Returns:
        Tuple[str, List[Dict], List[Dict]]: (updated_note, updated_entities, updated_dates)
    """
    current_note = note
    current_entities = [dict(e) for e in entities]
    current_dates = [dict(d) for d in dates]
    for r in relationships:
        if not r.get('diagnoses'):
            continue
        diag = r['diagnoses'][0]['diagnosis']
        date_key = r['date']
        e = next((e for e in current_entities if e['label'] == diag), None)
        d = next((d for d in current_dates if d['parsed'] == date_key), None)
        if not e or not d:
            continue
        gap = abs(d['start'] - e['end'])
        if gap >= min_distance:
            continue
        insert_pos = min(e['end'], d['start'])
        filler = " " + random.choice(filler_phrases) + " "
        current_note = current_note[:insert_pos] + filler + current_note[insert_pos:]
        delta = len(filler)
        current_entities = shift_spans_after_insertion(current_entities, insert_pos, delta)
        current_dates = shift_spans_after_insertion(current_dates, insert_pos, delta)
    return current_note, current_entities, current_dates


def add_filler_text(note: str, target_length: int, filler_phrases: List[str]) -> str:
    """
    Add filler text to a note to reach a target length.
    Args:
        note (str): The base note text.
        target_length (int): Desired length after adding filler.
        filler_phrases (List[str]): List of phrases to use as filler.
    Returns:
        str: Note text with filler added.
    """
    while len(note) < target_length:
        note += " " + random.choice(filler_phrases)
    return note[:target_length]


def inject_noise(note: str, typo_prob: float = 0.01, ambiguous_prob: float = 0.01) -> str:
    """
    Inject typos or ambiguous phrases into the note at a given probability.
    Args:
        note (str): The original note text.
        typo_prob (float): Probability of introducing a typo per character.
        ambiguous_prob (float): Probability of replacing a phrase with an ambiguous one.
    Returns:
        str: Note text with noise injected.
    """
    chars = list(note)
    for i in range(len(chars)):
        if random.random() < typo_prob and chars[i].isalpha():
            chars[i] = random.choice('abcdefghijklmnopqrstuvwxyz')
    return ''.join(chars)


def deduplicate_entities(entities: List[Dict]) -> List[Dict]:
    """
    Remove duplicate entities based on label and position.
    Args:
        entities (List[Dict]): List of entity dicts.
    Returns:
        List[Dict]: Deduplicated list of entities.
    """
    seen = set()
    deduped = []
    for e in entities:
        key = (e['label'], e['start'], e['end'])
        if key not in seen:
            seen.add(key)
            deduped.append(e)
    return deduped


def deduplicate_diagnoses(diagnoses: List[Dict]) -> List[Dict]:
    """
    Remove duplicate diagnoses based on diagnosis and position.
    Args:
        diagnoses (List[Dict]): List of diagnosis dicts.
    Returns:
        List[Dict]: Deduplicated list of diagnoses.
    """
    seen = set()
    deduped = []
    for d in diagnoses:
        key = (d['diagnosis'], d['position'])
        if key not in seen:
            seen.add(key)
            deduped.append(d)
    return deduped


def correct_entity_positions(note: str, entities: List[Dict]) -> List[Dict]:
    """
    Find and correct the positions of entities in the note.
    Args:
        note (str): The note text.
        entities (List[Dict]): List of entity dicts with 'label'.
    Returns:
        List[Dict]: Entities with corrected 'start' and 'end' positions.
    """
    corrected = []
    for e in entities:
        label = e['label']
        idx = note.find(label)
        if idx != -1:
            corrected.append({'label': label, 'start': idx, 'end': idx + len(label)})
    return corrected


def extract_dates_from_text(note: str) -> List[Tuple[str, int, int]]:
    """
    Extract simple date strings and positions from note text using basic regexes.
    Args:
        note (str): The note text.
    Returns:
        List[Tuple[str, int, int]]: (date_str, start, end)
    """
    patterns = [
        r"\(\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\)",
        r"\(\d{4}-\d{2}-\d{2}\)",
        r"\(\d{2}/\d{2}/\d{4}\)",
        r"\(\d{2}\.\d{2}\.\d{4}\)"
    ]
    matches = []
    for pat in patterns:
        for m in re.finditer(pat, note, flags=re.IGNORECASE):
            matches.append((m.group(0), m.start(), m.end()))
    return sorted(matches, key=lambda x: x[1])


def parse_date_str(date_str: str) -> Optional[str]:
    """
    Parse a simple date string into ISO format (YYYY-MM-DD). Returns None if unsupported.
    Args:
        date_str (str): Original date string as in text (with parentheses).
    Returns:
        Optional[str]: ISO date.
    """
    s = date_str.strip('()')
    m = re.match(r"(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})", s, flags=re.IGNORECASE)
    if m:
        day = int(m.group(1))
        month_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
        month = month_map[m.group(2).lower()[:3]]
        year = int(m.group(3))
        return f"{year:04d}-{month:02d}-{day:02d}"
    m = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.match(r"(\d{2})/(\d{2})/(\d{4})", s)
    if m:
        return f"{int(m.group(3)):04d}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
    m = re.match(r"(\d{2})\.(\d{2})\.(\d{4})", s)
    if m:
        return f"{int(m.group(3)):04d}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
    return None


def find_entity_all_positions(note: str, label: str) -> List[Tuple[int, int]]:
    """
    Find all occurrences of an entity label in the note (case-insensitive, spaces/underscores handled).
    Args:
        note (str): The note text.
        label (str): Entity label to search.
    Returns:
        List[Tuple[int, int]]: List of (start, end) positions.
    """
    positions = []
    term = label.replace('_', ' ')
    start = 0
    low_note = note.lower()
    low_term = term.lower()
    while True:
        idx = low_note.find(low_term, start)
        if idx == -1:
            break
        positions.append((idx, idx + len(term)))
        start = idx + 1
    if not positions and '_' in label:
        term2 = label
        start = 0
        low_term2 = term2.lower()
        while True:
            idx = low_note.find(low_term2, start)
            if idx == -1:
                break
            positions.append((idx, idx + len(term2)))
            start = idx + 1
    return positions


def correct_date_positions(note: str, dates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Correct date positions by matching their 'original' string in the note.
    Args:
        note (str): The note text.
        dates (List[Dict]): Date dicts with 'original','start','end'.
    Returns:
        List[Dict]: Dates with corrected 'start' and 'end' where possible.
    """
    corrected = []
    for d in dates:
        s = d.get('original', '')
        best = None
        best_dist = None
        for m in re.finditer(re.escape(s), note):
            pos = m.start()
            dist = abs(pos - d.get('start', pos))
            if best is None or dist < best_dist:
                best = (pos, m.end())
                best_dist = dist
        if best:
            corrected.append({'original': s, 'parsed': d.get('parsed'), 'start': best[0], 'end': best[1]})
        else:
            corrected.append(d)
    return corrected
