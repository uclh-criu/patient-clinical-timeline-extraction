import json

DATE_CUI = "410671006"


def load_medcat_export(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_valid_ann(a):
    # Now using MedCAT's "correct" flag, and excluding deleted ones
    return bool(a.get("correct", False)) and not a.get("deleted", False)


def get_validated_entities(doc, date_cui=DATE_CUI):
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


def get_validated_dates(doc, date_cui=DATE_CUI):
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


def get_validated_links(doc, date_cui=DATE_CUI):
    """
    Return date↔entity links as pairs of IDs from a single MedCAT document.
    Includes any created link (does NOT check relation.validated).
    Endpoints must be 'correct' and not 'deleted'; exactly one side must be a date.
    """
    # Build a quick index of validated anns so we can check types by id
    ann_map = {a["id"]: a for a in doc.get("annotations", []) if _is_valid_ann(a)}

    links = []
    for rel in doc.get("relations", []):
        s_id, e_id = rel.get("start_entity"), rel.get("end_entity")
        s = ann_map.get(s_id)
        e = ann_map.get(e_id)
        if not s or not e:
            continue

        is_date_s = s.get("cui") == date_cui
        is_date_e = e.get("cui") == date_cui
        if is_date_s ^ is_date_e:
            date_id = s_id if is_date_s else e_id
            ent_id = e_id if is_date_s else s_id
            links.append({"date_id": date_id, "entity_id": ent_id})

    return links


def doc_to_entities_json(ents):
    """JSON for entities (expects pre-filtered list of dicts)."""
    return json.dumps(ents, ensure_ascii=False)


def doc_to_dates_json(dates):
    """JSON for dates (expects pre-filtered list of dicts)."""
    return json.dumps(dates, ensure_ascii=False)


def doc_to_links_json(links):
    """JSON for links (expects pre-filtered list of dicts)."""
    return json.dumps(links, ensure_ascii=False)


def doc_to_links_value_json(links, id2value):
    """JSON for value pairs; uses provided {id: value} mapping."""
    links_value = [
        {"date": id2value.get(L["date_id"]), "entity": id2value.get(L["entity_id"])}
        for L in links
    ]
    return json.dumps(links_value, ensure_ascii=False)

def id2value_from_items(*item_lists):
    m = {}
    for lst in item_lists:
        for x in lst:
            m[x["id"]] = x.get("value", "")
    return m

def make_row(doc_id, note_text, entities_json, dates_json, links_json):
    return {
        "doc_id": doc_id,
        "note_text": note_text,
        "entities_json": entities_json,
        "dates_json": dates_json,
        "links_json": links_json,
    }
