import json

def load_medcat_export(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _is_valid_ann(a):
    # Now using MedCAT's "correct" flag, and excluding deleted ones
    return bool(a.get("correct", False)) and not a.get("deleted", False)

def get_validated_entities(doc, date_cui):
    """
    Return validated non-date entities from a single MedCAT document.
    """
    out = []
    for a in doc.get("annotations", []):
        if not _is_valid_ann(a):
            continue
        if a.get("cui") == date_cui:
            continue
        out.append({
            "id": a["id"],
            "value": a.get("value", ""),
            "cui": a.get("cui"),
            "start": a.get("start"),
            "end": a.get("end"),
        })
    return out

def get_validated_dates(doc, date_cui):
    """
    Return validated date annotations (raw) from a single MedCAT document.
    """
    out = []
    for a in doc.get("annotations", []):
        if not _is_valid_ann(a):
            continue
        if a.get("cui") != date_cui:
            continue
        out.append({
            "id": a["id"],
            "value": a.get("value", ""),
            "start": a.get("start"),
            "end": a.get("end"),
        })
    return out

def get_validated_relative_dates(doc, relative_date_cui):
    """
    Return validated relative date annotations from a single MedCAT document.
    """
    out = []
    for a in doc.get("annotations", []):
        if not _is_valid_ann(a):
            continue
        if a.get("cui") != relative_date_cui:
            continue
        out.append({
            "id": a["id"],
            "value": a.get("value", ""),
            "start": a.get("start"),
            "end": a.get("end"),
        })
    return out

def get_validated_relations(doc, date_cui, relative_date_cui):
    """
    Return date↔entity relations as pairs of IDs from a single MedCAT document.
    Includes any created relation (does NOT check relation.validated).
    Endpoints must be 'correct' and not 'deleted'; exactly one side must be a date (absolute or relative).
    """
    # Build a quick index of validated anns so we can check types by id
    ann_map = {a["id"]: a for a in doc.get("annotations", []) if _is_valid_ann(a)}

    relations = []
    for rel in doc.get("relations", []):
        s_id, e_id = rel.get("start_entity"), rel.get("end_entity")
        s = ann_map.get(s_id)
        e = ann_map.get(e_id)
        if not s or not e:
            continue

        is_date_s = s.get("cui") == date_cui
        is_date_e = e.get("cui") == date_cui
        is_rel_date_s = s.get("cui") == relative_date_cui
        is_rel_date_e = e.get("cui") == relative_date_cui
        
        # Check if exactly one side is a date (absolute or relative)
        if (is_date_s or is_rel_date_s) ^ (is_date_e or is_rel_date_e):
            date_id = s_id if (is_date_s or is_rel_date_s) else e_id
            ent_id = e_id if (is_date_s or is_rel_date_s) else s_id
            relations.append({"date_id": date_id, "entity_id": ent_id})

    return relations

def doc_to_entities_json(ents):
    """JSON for entities (expects pre-filtered list of dicts)."""
    return json.dumps(ents, ensure_ascii=False)

def doc_to_dates_json(dates):
    """JSON for dates (expects pre-filtered list of dicts)."""
    return json.dumps(dates, ensure_ascii=False)

def doc_to_relative_dates_json(relative_dates):
    """JSON for relative dates (expects pre-filtered list of dicts)."""
    return json.dumps(relative_dates, ensure_ascii=False)

def doc_to_relations_json(relations):
    """JSON for relations (expects pre-filtered list of dicts)."""
    return json.dumps(relations, ensure_ascii=False)

def doc_to_relations_value_json(relations, id2value):
    """JSON for value pairs; uses provided {id: value} mapping."""
    relations_value = [
        {
            "date": id2value.get(L["date_id"]), 
            "entity": id2value.get(L["entity_id"]),
            "date_id": L["date_id"],
            "entity_id": L["entity_id"]
        }
        for L in relations
    ]
    return json.dumps(relations_value, ensure_ascii=False)

def id2value_from_items(*item_lists):
    m = {}
    for lst in item_lists:
        for x in lst:
            m[x["id"]] = x.get("value", "")
    return m

def make_row(doc_id, note_text, entities_json, dates_json, relative_dates_json, relations_json):
    return {
        "doc_id": doc_id,
        "note_text": note_text,
        "entities_json": entities_json,
        "dates_json": dates_json,
        "relative_dates_json": relative_dates_json,
        "relations_json": relations_json,
    }
