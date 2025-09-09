import json
import pandas as pd

DATE_CUI = "410671006"

# Load MedCAT JSON
with open("./data/MedCAT_Export_With_Text_2025-09-09_10_18_25.json", "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for project in data["projects"]:
    for doc in project["documents"]:
        doc_id = doc["id"]
        text = doc["text"]

        # All annotations in this doc
        anns = {a["id"]: a for a in doc.get("annotations", [])}

        # Collect dates and non-dates
        dates = [a for a in anns.values() if a.get("cui") == DATE_CUI]
        others = [a for a in anns.values() if a.get("cui") != DATE_CUI]

        # Relations explicitly annotated as 'link'
        link_pairs = set()
        for rel in doc.get("relations", []):
            link_pairs.add(tuple(sorted([rel["start_entity"], rel["end_entity"]])))

        # Create entity–date pairs
        for date in dates:
            for ent in others:
                ent1, ent2 = date, ent

                ent1_id, ent2_id = ent1["id"], ent2["id"]
                ent1_val, ent2_val = ent1["value"], ent2["value"]
                ent1_cui, ent2_cui = ent1.get("cui"), ent2.get("cui")
                ent1_s, ent1_e = ent1.get("start"), ent1.get("end")
                ent2_s, ent2_e = ent2.get("start"), ent2.get("end")

                # Determine label
                if tuple(sorted([ent1_id, ent2_id])) in link_pairs:
                    label, label_id = "LINK", 1
                else:
                    label, label_id = "NO_LINK", 0

                # Insert ADE-style markers into text
                def insert_marker(txt, start, end, tag_open, tag_close):
                    return txt[:start] + tag_open + txt[start:end] + tag_close + txt[end:]

                if ent1_s is not None and ent2_s is not None:
                    if ent1_s < ent2_s:
                        marked = insert_marker(text, ent2_s, ent2_e, "[s2]", "[e2]")
                        marked = insert_marker(marked, ent1_s, ent1_e, "[s1]", "[e1]")
                    else:
                        marked = insert_marker(text, ent1_s, ent1_e, "[s2]", "[e2]")
                        marked = insert_marker(marked, ent2_s, ent2_e, "[s1]", "[e1]")
                else:
                    marked = text

                rows.append({
                    "relation_token_span_ids": None,
                    "ent1_ent2_start": (ent1_s, ent2_s),
                    "ent1": ent1_val,
                    "ent2": ent2_val,
                    "label": label,
                    "label_id": label_id,
                    "ent1_type": "DATE",
                    "ent2_type": "ENTITY",
                    "ent1_id": ent1_id,
                    "ent2_id": ent2_id,
                    "ent1_cui": ent1_cui,
                    "ent2_cui": ent2_cui,
                    "doc_id": doc_id,
                    "text": marked
                })

# Save TSV
df = pd.DataFrame(rows)
df.to_csv("./data/relations_date_entity.tsv", sep="\t", index=False)
print(f"Generated {len(df)} date-entity relations")
print("Label distribution:")
print(df["label"].value_counts())
print("\nFirst few rows:")
print(df)