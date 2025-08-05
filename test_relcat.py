from medcat.rel_cat import RelCAT
import json
from pprint import pprint

import os
try:
    rc_path = "models/ade_relcat_model"
    relCAT = RelCAT.load(rc_path)
except Exception as e:
    raise ValueError(
        f"Unable to load RelCAT model from '{rc_path}'. "
        f"Exists? {os.path.exists(rc_path)}; isdir? {os.path.isdir(rc_path)} "
        f"Stuff in dir: {os.listdir(rc_path) if os.path.isdir(rc_path) else ['NOT A DIR']}")

docs_with_anns = {"text": "REASON FOR CONSULTATION: , Left hip fracture.,HISTORY OF PRESENT ILLNESS: , The patient is a pleasant 53-year-old female with a known history of sciatica, apparently presented to the emergency room due to severe pain in the left lower extremity and unable to bear weight.  History was obtained from the patient.  As per the history, she reported that she has been having back pain with left leg pain since past 4 weeks.  She has been using a walker for ambulation due to disabling pain in her left thigh and lower back.  She was seen by her primary care physician and was scheduled to go for MRI yesterday.  However, she was walking and her right foot got caught on some type of rug leading to place excessive weight on her left lower extremity to prevent her fall.  Since then, she was unable to ambulate.  The patient called paramedics and was brought to the emergency room.  She denied any history of fall.  She reported that she stepped the wrong way causing the pain to become worse.  She is complaining of severe pain in her lower extremity and back pain.  Denies any tingling or numbness.  Denies any neurological symptoms.  Denies any bowel or bladder incontinence.,X-rays were obtained which were remarkable for left hip fracture.  Orthopedic consultation was called for further evaluation and management.  On further interview with the patient, it is noted that she has a history of malignant melanoma, which was diagnosed approximately 4 to 5 years ago.  She underwent surgery at that time and subsequently, she was noted to have a spread to the lymphatic system and lymph nodes for which she underwent surgery in 3/2008.,PAST MEDICAL HISTORY: , Sciatica and melanoma.,PAST SURGICAL HISTORY:  ,As discussed above, surgery for melanoma and hysterectomy.,ALLERGIES: , NONE.,SOCIAL HISTORY: , Denies any tobacco or alcohol use.  She is divorced with 2 children.  She lives with her son.,PHYSICAL EXAMINATION:,GENERAL:  The patient is well developed, well nourished in mild distress secondary to left lower extremity and back pain.,MUSCULOSKELETAL:  Examination of the left lower extremity, there is presence of apparent shortening and external rotation deformity.  Tenderness to palpation is present.  Leg rolling is positive for severe pain in the left proximal hip.  Further examination of the spine is incomplete secondary to severe leg pain.  She is unable to perform a straight leg raising.  EHL/EDL 5/5.  2+ pulses are present distally.  Calf is soft and nontender.  Homans sign is negative.  Sensation to light touch is intact.,IMAGING:,  AP view of the hip is reviewed.  Only 1 limited view is obtained.  This is a poor quality x-ray with a lot of soft tissue shadow.  This x-ray is significant for basicervical-type femoral neck fracture.  Lesser trochanter is intact.  This is a high intertrochanteric fracture/basicervical.  There is presence of lytic lesion around the femoral neck, which is not well delineated on this particular x-ray.  We need to order repeat x-rays including AP pelvis, femur, and knee.,LABS:,  Have been reviewed.,ASSESSMENT: , The patient is a 53-year-old female with probable pathological fracture of the left proximal femur.,DISCUSSION AND PLAN: , Nature and course of the diagnosis has been discussed with the patient.  Based on her presentation without any history of obvious fall or trauma and past history of malignant melanoma, this appears to be a pathological fracture of the left proximal hip.  At the present time, I would recommend obtaining a bone scan and repeat x-rays, which will include AP pelvis, femur, hip including knee.  She denies any pain elsewhere.  She does have a past history of back pain and sciatica, but at the present time, this appears to be a metastatic bone lesion with pathological fracture.  I have discussed the case with Dr. X and recommended oncology consultation.,With the above fracture and presentation, she needs a left hip hemiarthroplasty versus calcar hemiarthroplasty, cemented type.  Indication, risk, and benefits of left hip hemiarthroplasty has been discussed with the patient, which includes, but not limited to bleeding, infection, nerve injury, blood vessel injury, dislocation early and late, persistent pain, leg length discrepancy, myositis ossificans, intraoperative fracture, prosthetic fracture, need for conversion to total hip replacement surgery, revision surgery, DVT, pulmonary embolism, risk of anesthesia, need for blood transfusion, and cardiac arrest.  She understands above and is willing to undergo further procedure.  The goal and the functional outcome have been explained.  Further plan will be discussed with her once we obtain the bone scan and the radiographic studies.  We will also await for the oncology feedback and clearance.,Thank you very much for allowing me to participate in the care of this patient.  I will continue to follow up.",
                   "annotations": [{
                            "id": 1011,
                            "user": "admin",
                            "cui": "161432005",
                            "value": "history of malignant melanoma",
                            "start": 1382,
                            "end": 1411,
  
                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1012,
                            "user": "admin",
                            "cui": "161432005",
                            "value": "history of malignant melanoma",
                            "start": 3347,
                            "end": 3376,
                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1013,
                            "user": "admin",
                            "cui": "52734007",
                            "value": "total hip replacement surgery",
                            "start": 4323,
                            "end": 4352,

                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1014,
                            "user": "admin",
                            "cui": "127287001",
                            "value": "intertrochanteric fracture",
                            "start": 2802,
                            "end": 2828,
   
                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1015,
                            "user": "admin",
                            "cui": "213270002",
                            "value": "intraoperative fracture",
                            "start": 4254,
                            "end": 4277,

                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1016,
                            "user": "admin",
                            "cui": "446050000",
                            "value": "primary care physician",
                            "start": 541,
                            "end": 563,
           
                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1017,
                            "user": "admin",
                            "cui": "5913000",
                            "value": "femoral neck fracture",
                            "start": 2733,
                            "end": 2754,

                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1018,
                            "user": "admin",
                            "cui": "268029009",
                            "value": "pathological fracture",
                            "start": 3120,
                            "end": 3141,
  
                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1019,
                            "user": "admin",
                            "cui": "268029009",
                            "value": "pathological fracture",
                            "start": 3399,
                            "end": 3420,

                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1020,
                            "user": "admin",
                            "cui": "268029009",
                            "value": "pathological fracture",
                            "start": 3748,
                            "end": 3769,
   
                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1021,
                            "user": "admin",
                            "cui": "32153003",
                            "value": "left lower extremity",
                            "start": 224,
                            "end": 244,

                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1022,
                            "user": "admin",
                            "cui": "32153003",
                            "value": "left lower extremity",
                            "start": 724,
                            "end": 744,

                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1023,
                            "user": "admin",
                            "cui": "165232002",
                            "value": "bladder incontinence",
                            "start": 1152,
                            "end": 1172,

                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1024,
                            "user": "admin",
                            "cui": "5880005",
                            "value": "PHYSICAL EXAMINATION",
                            "start": 1895,
                            "end": 1915,

                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1025,
                            "user": "admin",
                            "cui": "32153003",
                            "value": "left lower extremity",
                            "start": 2003,
                            "end": 2023,

                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1026,
                            "user": "admin",
                            "cui": "32153003",
                            "value": "left lower extremity",
                            "start": 2076,
                            "end": 2096,

                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1027,
                            "user": "admin",
                            "cui": "57662003",
                            "value": "injury, blood vessel",
                            "start": 4135,
                            "end": 4155,

                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1028,
                            "user": "admin",
                            "cui": "44551007",
                            "value": "myositis ossificans",
                            "start": 4233,
                            "end": 4252,

                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1029,
                            "user": "admin",
                            "cui": "59282003",
                            "value": "pulmonary embolism",
                            "start": 4377,
                            "end": 4395,

                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1030,
                            "user": "admin",
                            "cui": "116859006",
                            "value": "blood transfusion",
                            "start": 4426,
                            "end": 4443,

                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1031,
                            "user": "admin",
                            "cui": "224994002",
                            "value": "excessive weight",
                            "start": 700,
                            "end": 716,

                            "meta_anns": {}
                        },
                        {
                            "id": 1032,
                            "user": "admin",
                            "cui": "89890002",
                            "value": "lymphatic system",
                            "start": 1557,
                            "end": 1573,

                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1033,
                            "user": "admin",
                            "cui": "261554009",
                            "value": "revision surgery",
                            "start": 4354,
                            "end": 4370,

                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1034,
                            "user": "admin",
                            "cui": "428942009",
                            "value": "history of fall",
                            "start": 893,
                            "end": 908,

                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1035,
                            "user": "admin",
                            "cui": "61685007",
                            "value": "lower extremity",
                            "start": 1031,
                            "end": 1046,
   
                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1036,
                            "user": "admin",
                            "cui": "392521001",
                            "value": "MEDICAL HISTORY",
                            "start": 1638,
                            "end": 1653,

                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1037,
                            "user": "admin",
                            "cui": "106028002",
                            "value": "MUSCULOSKELETAL",
                            "start": 2039,
                            "end": 2054,

                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1038,
                            "user": "admin",
                            "cui": "417662000",
                            "value": "past history of",
                            "start": 3634,
                            "end": 3649,

                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1039,
                            "user": "admin",
                            "cui": "225728007",
                            "value": "emergency room",
                            "start": 183,
                            "end": 197,
                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1040,
                            "user": "admin",
                            "cui": "225728007",
                            "value": "emergency room",
                            "start": 861,
                            "end": 875,
                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1041,
                            "user": "admin",
                            "cui": "160476009",
                            "value": "SOCIAL HISTORY",
                            "start": 1783,
                            "end": 1797,
                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1042,
                            "user": "admin",
                            "cui": "248324001",
                            "value": "well nourished",
                            "start": 1958,
                            "end": 1972,
                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1043,
                            "user": "admin",
                            "cui": "244696009",
                            "value": "proximal femur",
                            "start": 3154,
                            "end": 3168,
                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1044,
                            "user": "admin",
                            "cui": "410429000",
                            "value": "cardiac arrest",
                            "start": 4449,
                            "end": 4463,
                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1045,
                            "user": "admin",
                            "cui": "287047008",
                            "value": "left leg pain",
                            "start": 386,
                            "end": 399,
                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1046,
                            "user": "admin",
                            "cui": "161891005",
                            "value": "and back pain",
                            "start": 1047,
                            "end": 1060,
                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1047,
                            "user": "admin",
                            "cui": "26175008",
                            "value": "approximately",
                            "start": 1433,
                            "end": 1446,
                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1048,
                            "user": "admin",
                            "cui": "161891005",
                            "value": "and back pain",
                            "start": 2024,
                            "end": 2037,

                            "acc": 1.0,
                            "meta_anns": {}
                        },
                        {
                            "id": 1049,
                            "user": "admin",
                            "cui": "1199008",
                            "value": "neurological",
                            "start": 1108,
                            "end": 1120
                        }
                        ]}

output_doc_with_relations = relCAT.predict_text_with_anns(text=docs_with_anns["text"], annotations=docs_with_anns["annotations"])



# Extract relations in a more readable format
relations = output_doc_with_relations._.relations
relations_by_type = {}

print("\n===== FORMATTED RELATIONS =====\n")
print(f"Total relations found: {len(relations)}\n")

# Organize relations by relation type
for relation in relations:
    rel_type = relation['relation']
    if rel_type not in relations_by_type:
        relations_by_type[rel_type] = []
    relations_by_type[rel_type].append(relation)

# Print relations by type
for rel_type, rel_list in relations_by_type.items():
    print(f"\n== {rel_type} Relations ({len(rel_list)}) ==\n")
    
    # Sort by confidence score (highest first)
    rel_list.sort(key=lambda x: x['confidence'], reverse=True)
    
    for i, rel in enumerate(rel_list, 1):
        print(f"  {i}. {rel['ent1_text']} → {rel['ent2_text']}")
        print(f"     Confidence: {rel['confidence']:.2f}")
        if i < len(rel_list):  # Don't print a separator after the last item
            print("     ---")
    
# Print highest confidence relations overall
print("\n== Top 10 Relations by Confidence ==\n")
top_relations = sorted(relations, key=lambda x: x['confidence'], reverse=True)[:10]
for i, rel in enumerate(top_relations, 1):
    print(f"  {i}. {rel['relation']}: {rel['ent1_text']} → {rel['ent2_text']}")
    print(f"     Confidence: {rel['confidence']:.2f}")
    if i < len(top_relations):  # Don't print a separator after the last item
        print("     ---")