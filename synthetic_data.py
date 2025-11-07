from openai import OpenAI
import json

# ------------------------------
# CONFIGURATION
# ------------------------------
API_KEY = "*"
NUM_PEOPLE = 100
NUM_JOBS = 50
NUM_COMPANIES = 30
OUTPUT_FILE = "graph_dataset.json"

client = OpenAI(api_key=API_KEY)

prompt = f"""
Generate a JSON dataset for a homogeneous graph neural network with three node types:
1. person
2. job_posting
3. company

SCHEMA:

PERSON NODE:
- id: string (e.g., "P001")
- type: "person"
- name: realistic full name
- location: city, state
- gpa: float
- education: list of objects {{"degree": string, "institution": string, "graduation_year": int, "gpa": float}}
- work_experience: list of objects {{"company": string, "title": string, "years": float, "description": string}}
- skills: list of strings
- soft_skills: list of strings
- certifications: list of strings
- projects: list of objects {{"title": string, "description": string, "technologies": list of strings}}
- resume_summary: string

JOB_POSTING NODE:
- id: string (e.g., "J001")
- type: "job_posting"
- title: string
- company_id: string
- location: city, state
- salary_range: string
- skills_required: list of strings
- description: string
- employment_type: string
- date_posted: string (YYYY-MM-DD)
- experience_level: string

COMPANY NODE:
- id: string (e.g., "C001")
- type: "company"
- name: string
- industry: string
- size: string
- location: city, state
- founded: int
- description: string

EDGES:
- Format: {{"source": string, "target": string, "relation": string}}
- Relations: "worked_at", "applied_to", "posted_by"

REQUIREMENTS:
- Create {NUM_PEOPLE} person nodes, {NUM_JOBS} job_posting nodes, and {NUM_COMPANIES} company nodes.
- IDs must be unique and consistent across nodes and edges.
- All job postings must reference valid company_ids.
- Edges must logically connect people, jobs, and companies.
- Output ONLY valid JSON with two top-level keys: "nodes" (list) and "edges" (list).
"""

print("Generating dataset... This may take a few seconds.")
response = client.responses.create(
    model="gpt-4o-mini",
    input=prompt,
    temperature=0.7
)

dataset_text = response.output_text
try:
    dataset_text = response.output_text.strip()

    if dataset_text.startswith("```json"):
        dataset_text = dataset_text[len("```json"):]

    if dataset_text.endswith("```"):
        dataset_text = dataset_text[:-3]

    dataset_text = dataset_text.strip()

    dataset = json.loads(dataset_text)
except json.JSONDecodeError as e:
    print("Error parsing JSON:", e)
    print("Raw output:\n", dataset_text)
    exit()

with open(OUTPUT_FILE, "w") as f:
    json.dump(dataset, f, indent=2)

print(f"Dataset generated and saved to {OUTPUT_FILE}")
