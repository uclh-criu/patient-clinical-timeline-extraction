import random
import datetime
import json
import re
from faker import Faker

fake = Faker()

# Define possible diagnoses
DIAGNOSES = [
    "migraine", "tension headache", "cluster headache",
    "pituitary adenoma", "prolactinoma", "macroadenoma", "microadenoma",
    "diabetes mellitus", "hypertension", "hyperlipidemia",
    "asthma", "COPD", "pneumonia", "bronchitis",
    "depression", "anxiety", "bipolar disorder", "schizophrenia",
    "osteoarthritis", "rheumatoid arthritis", "gout", "fibromyalgia",
    "hypothyroidism", "hyperthyroidism", "adrenal insufficiency", "Cushing's syndrome",
    "IBS", "Crohn's disease", "ulcerative colitis", "GERD",
    "multiple sclerosis", "Parkinson's disease", "epilepsy", "stroke",
    "anemia", "leukemia", "lymphoma", "myeloma"
]

# Templates for clinical note sections
SECTION_TEMPLATES = [
    "{visit_type}(({date})[date]): pt presents with {symptom}. {assessment}",
    "{specialty} CONSULT ({date})[date]: {assessment} {plan}",
    "F/U ({date})[date]: {status}. {assessment} {plan}",
    "URGENT REVIEW ({date})[date]: {symptom}. {assessment}",
    "Labs ({date})[date]: {lab_results}. {assessment}",
    "Phone note(({date})[date]): {status}. {plan}",
    "{imaging_type} ({date})[date]: {imaging_results}. imp: {diagnosis}[diagnosis]",
    "CLINIC VISIT ({date})[date]: {symptom} {status} {assessment} {plan}"
]

VISIT_TYPES = ["Initial assessment", "New patient", "Evaluation", "Consult", "Visit"]
SPECIALTIES = ["Neurology", "Endocrinology", "Cardiology", "Internal Medicine", "Oncology", "Psychiatry", "Rheumatology"]
SYMPTOMS = [
    "headache x{duration}", "dizziness", "fatigue", "weight loss", "weight gain", 
    "visual changes", "nausea/vomiting", "chest pain", "shortness of breath",
    "joint pain", "abdominal pain", "back pain", "mood changes", "insomnia",
    "excessive thirst", "frequent urination", "fever", "cough", "rash"
]
STATUS = [
    "improved", "slightly improved", "unchanged", "worsening", "resolved", 
    "stable", "fluctuating", "progressive", "remission", "relapse"
]
PLANS = [
    "continue current medications", "increase dose", "taper medication", 
    "switch to {medication}", "refer to {specialty}", "schedule {imaging_type}", 
    "follow up in {follow_up_time}", "admit to hospital", "observe and reassess"
]
ASSESSMENTS = [
    "consistent with {diagnosis}[diagnosis]", 
    "suspect {diagnosis}[diagnosis]", 
    "likely {diagnosis}[diagnosis]", 
    "rule out {diagnosis}[diagnosis]", 
    "unclear etiology, could be {diagnosis}[diagnosis] vs {diagnosis2}[diagnosis]",
    "confirmed {diagnosis}[diagnosis]",
    "new diagnosis of {diagnosis}[diagnosis]",
    "resolving {diagnosis}[diagnosis]"
]
IMAGING_TYPES = ["MRI", "CT", "X-ray", "Ultrasound", "PET scan"]
IMAGING_RESULTS = [
    "no significant findings", 
    "shows {size}cm mass in {location}", 
    "reveals {diagnosis}[diagnosis]", 
    "consistent with {diagnosis}[diagnosis]",
    "evidence of {pathology}"
]
LAB_RESULTS = [
    "WBC elevated", "anemia", "elevated glucose", "low TSH", "high cortisol", 
    "elevated prolactin", "abnormal LFTs", "positive ANA", "elevated CRP",
    "low vitamin D", "normal CBC", "normal chemistry panel"
]
PATHOLOGIES = [
    "inflammation", "edema", "atrophy", "lesion", "mass effect", 
    "enlargement", "stenosis", "fluid collection", "hemorrhage"
]
MEDICATIONS = [
    "prednisone", "levothyroxine", "metformin", "lisinopril", "atorvastatin",
    "sumatriptan", "cabergoline", "topiramate", "amitriptyline", "fluoxetine",
    "hydrochlorothisone", "insulin", "aspirin", "ibuprofen", "acetaminophen"
]
LOCATIONS = [
    "pituitary", "brain", "lung", "liver", "kidney", "thyroid", 
    "adrenal gland", "pancreas", "colon", "stomach", "spine"
]
DURATIONS = ["1 day", "3 days", "1 week", "2 weeks", "1 month", "3 months", "6 months", "1 year"]
FOLLOW_UP_TIMES = ["1 week", "2 weeks", "1 month", "3 months", "6 months"]
SIZES = ["0.5", "1.2", "2.3", "3.1", "4.5", "0.8", "1.5", "2.7"]

def generate_date_formats(date):
    """Generate date in various formats"""
    formats = [
        date.strftime("%d-%m-%Y"),          # 15-01-2023
        date.strftime("%d.%m.%y"),          # 15.01.23
        date.strftime("%d/%m/%y"),          # 15/01/23
        date.strftime("%d/%m/%Y"),          # 15/01/2023
        date.strftime("%Y-%m-%d"),          # 2023-01-15
        date.strftime("%d %b %Y"),          # 15 Jan 2023
        date.strftime("%d %b'%y"),          # 15 Jan'23
        date.strftime("%dst %b %Y").replace("1st", "1st").replace("2st", "2nd").replace("3st", "3rd"),  # 1st Jan 2023
        date.strftime("%dnd %b %Y").replace("1nd", "1st").replace("2nd", "2nd").replace("3nd", "3rd"),  # 2nd Jan 2023
        date.strftime("%drd %b %Y").replace("1rd", "1st").replace("2rd", "2nd").replace("3rd", "3rd"),  # 3rd Jan 2023
        date.strftime("%dth %b %Y").replace("1th", "1st").replace("2th", "2nd").replace("3th", "3rd")   # 4th, 5th, etc.
    ]
    return random.choice(formats)

def generate_section(previous_date=None, min_days=1, max_days=60):
    """Generate a clinical note section"""
    if previous_date:
        # Generate a date after the previous date
        days_after = random.randint(min_days, max_days)
        date = previous_date + datetime.timedelta(days=days_after)
    else:
        # Start with a random date in the past year
        days_ago = random.randint(30, 365)
        date = datetime.datetime.now() - datetime.timedelta(days=days_ago)
    
    date_str = generate_date_formats(date)
    
    template = random.choice(SECTION_TEMPLATES)
    
    # Fill in placeholders
    visit_type = random.choice(VISIT_TYPES)
    specialty = random.choice(SPECIALTIES)
    symptom = random.choice(SYMPTOMS).replace("{duration}", random.choice(DURATIONS))
    status = random.choice(STATUS)
    
    # Generate a primary diagnosis for this section
    diagnosis = random.choice(DIAGNOSES)
    diagnosis2 = random.choice([d for d in DIAGNOSES if d != diagnosis])  # Different from primary

    # Format diagnoses with underscores
    diagnosis_fmt = diagnosis.replace(' ', '_')
    diagnosis2_fmt = diagnosis2.replace(' ', '_')

    assessment = random.choice(ASSESSMENTS).replace(
        "{diagnosis}", diagnosis_fmt  # Use formatted diagnosis
    ).replace(
        "{diagnosis2}", diagnosis2_fmt # Use formatted diagnosis2
    )
    
    plan = random.choice(PLANS).replace(
        "{medication}", random.choice(MEDICATIONS)
    ).replace(
        "{specialty}", random.choice(SPECIALTIES)
    ).replace(
        "{imaging_type}", random.choice(IMAGING_TYPES)
    ).replace(
        "{follow_up_time}", random.choice(FOLLOW_UP_TIMES)
    )
    
    imaging_type = random.choice(IMAGING_TYPES)
    imaging_results = random.choice(IMAGING_RESULTS).replace(
        "{size}", random.choice(SIZES)
    ).replace(
        "{location}", random.choice(LOCATIONS)
    ).replace(
        "{diagnosis}", diagnosis_fmt # Use formatted diagnosis
    ).replace(
        "{pathology}", random.choice(PATHOLOGIES)
    )
    
    lab_results = random.choice(LAB_RESULTS)
    
    section = template.format(
        visit_type=visit_type,
        date=date_str,
        specialty=specialty,
        symptom=symptom,
        status=status,
        assessment=assessment, # This now contains _fmt diagnoses
        plan=plan,
        imaging_type=imaging_type,
        imaging_results=imaging_results, # This now contains _fmt diagnoses
        lab_results=lab_results,
        diagnosis=diagnosis # This diagnosis placeholder in the original template might still need formatting if used directly
                            # Let's reformat the section string *after* initial format to be safe
    )
    
    # Ensure diagnosis tags in the final section string use underscore format
    # This replaces any remaining instances that might not have been caught by assessment/imaging formatting
    section = section.replace(diagnosis + '[diagnosis]', diagnosis_fmt + '[diagnosis]')
    section = section.replace(diagnosis + '[dx]', diagnosis_fmt + '[dx]')
    section = section.replace(diagnosis2 + '[diagnosis]', diagnosis2_fmt + '[diagnosis]')
    section = section.replace(diagnosis2 + '[dx]', diagnosis2_fmt + '[dx]')
    # Handle potential space in diagnosis tag
    section = section.replace(diagnosis + '[diagno sis]', diagnosis_fmt + '[diagno sis]')
    section = section.replace(diagnosis2 + '[diagno sis]', diagnosis2_fmt + '[diagno sis]')

    # Store the ground truth relationship using the underscore format
    ground_truth = {
        "date": date.strftime("%Y-%m-%d"),
        "date_position": section.find("[date]") - len(date_str), # Recalculate position if needed, though date format unchanged
        "diagnoses": []
    }

    # Find all diagnoses (underscore format) in the section
    # Regex to find underscore-formatted diagnosis followed by tag
    # \w+ matches letters, numbers, AND underscores
    for match in re.finditer(r'(\w+)\[(?:dx|diagnosis|diagno\s*sis)\]', section):
        found_diag = match.group(1) # This will be underscore_formatted
        position = match.start(1)   # Position of the diagnosis itself
        # Check if this diagnosis is one of the intended ones for this section (optional but good check)
        if found_diag == diagnosis_fmt or found_diag == diagnosis2_fmt:
             ground_truth["diagnoses"].append({
                 "diagnosis": found_diag.lower(), # Store lowercase underscore version
                 "position": position
             })
    
    return section, date, ground_truth

def generate_clinical_note(num_sections=5):
    """Generate a complete clinical note with multiple sections"""
    sections = []
    ground_truth = []
    
    date = None
    
    for i in range(num_sections):
        section, date, section_truth = generate_section(date)
        sections.append(section)
        ground_truth.append(section_truth)
    
    # Occasional typos and formatting issues for realism
    if random.random() < 0.3:
        # Add some typos
        note = "\n\n".join(sections)
        typo_chars = [",", ".", " ", "", ";"]
        for _ in range(random.randint(1, 5)):
            pos = random.randint(0, len(note) - 1)
            note = note[:pos] + random.choice(typo_chars) + note[pos:]
    else:
        note = "\n\n".join(sections)
    
    # Add some common abbreviations
    abbreviations = {
        "patient": "pt",
        "history": "hx",
        "diagnosis": "dx",
        "treatment": "tx",
        "with": "w/",
        "without": "w/o",
        "follow up": "f/u",
        "review": "r/v",
        "continue": "cont",
        "prescription": "rx"
    }
    
    for word, abbr in abbreviations.items():
        if random.random() < 0.7:  # 70% chance to use abbreviation
            note = note.replace(word, abbr)
    
    # Occasionally add spacing issues
    if random.random() < 0.4:
        note = note.replace(". ", ".")
    if random.random() < 0.3:
        note = note.replace(", ", ",")
    
    return note, ground_truth

def generate_dataset(num_notes=100):
    """Generate a dataset of clinical notes with ground truth relationships"""
    dataset = []
    
    for i in range(num_notes):
        note, ground_truth = generate_clinical_note(num_sections=random.randint(3, 8))
        
        # Create entry with note and ground truth
        entry = {
            "id": i,
            "clinical_note": note,
            "ground_truth": ground_truth
        }
        
        dataset.append(entry)
    
    return dataset

if __name__ == "__main__":
    # Generate dataset
    dataset = generate_dataset(100)
    
    # Save to file
    output_path = 'data/synthetic_data.json'
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated {len(dataset)} synthetic clinical notes with ground truth relationships to {output_path}")
    
    # Print a sample
    sample_idx = random.randint(0, len(dataset) - 1)
    sample = dataset[sample_idx]
    print("\nSample clinical note (ID:", sample_idx, "):")
    print(sample["clinical_note"])
    print("\nGround truth:")
    for section in sample["ground_truth"]:
        print(f"Date: {section['date']}")
        for diag in section["diagnoses"]:
            print(f"  - {diag['diagnosis']}") 