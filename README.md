##Graph Dataset Generator

This project uses the OpenAI API to automatically generate a synthetic graph dataset containing three node types — person, job_posting, and company — along with relational edges. The resulting dataset can be used for experiments in graph neural networks (GNNs), entity–relationship modeling, and recommendation systems.

###Overview

The generator produces structured JSON data that models a small ecosystem of individuals, job postings, and companies.
It ensures unique identifiers, valid cross-references, and realistic attribute values across all entities.

###Each dataset includes:
* Person nodes with education, experience, and skill attributes
* Job posting nodes with requirements and metadata
* Company nodes with descriptive and organizational information
* Edges connecting the nodes through logical relationships such as worked_at, applied_to, and posted_by

###Features
* Generates realistic, structured, and internally consistent JSON data
* Customizable number of people, jobs, and companies
* Validates JSON output before saving
* Ready for downstream tasks such as GNN training, entity-link prediction, or graph visualization

###Output Format
The output file contains two top-level keys: nodes and edges.

Example:

```json
{
  "nodes": [
    {"id": "P001", "type": "person", "name": "Jane Doe", "skills": ["Python", "SQL"]},
    {"id": "J001", "type": "job_posting", "title": "Data Scientist", "company_id": "C001"},
    {"id": "C001", "type": "company", "name": "DataForge AI"}
  ],
  "edges": [
    {"source": "P001", "target": "J001", "relation": "applied_to"},
    {"source": "P001", "target": "C001", "relation": "worked_at"},
    {"source": "C001", "target": "J001", "relation": "posted_by"}
  ]
}

