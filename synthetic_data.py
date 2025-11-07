from openai import OpenAI
import json
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GCNConv
import random

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
Generate a JSON dataset for a heterogeneous graph with three node types:
1. person
2. job_posting
3. company
...
(keep your original prompt here)
...
"""

response = client.responses.create(
    model="gpt-4o-mini",
    input=prompt,
    temperature=0.7
)

dataset_text = response.output_text.strip()

if dataset_text.startswith("```json"):
    dataset_text = dataset_text[len("```json"):]

if dataset_text.endswith("```"):
    dataset_text = dataset_text[:-3]

dataset = json.loads(dataset_text)

with open(OUTPUT_FILE, "w") as f:
    json.dump(dataset, f, indent=2)

print(f"Dataset saved to {OUTPUT_FILE}")


data = HeteroData()

person_ids = [p["id"] for p in dataset["nodes"] if p["type"] == "person"]
job_ids = [j["id"] for j in dataset["nodes"] if j["type"] == "job_posting"]
company_ids = [c["id"] for c in dataset["nodes"] if c["type"] == "company"]

# encode GPA for person, random for jobs/companies
data["person"].x = torch.tensor([ [p.get("gpa", 0.0)] for p in dataset["nodes"] if p["type"] == "person"], dtype=torch.float)
data["job_posting"].x = torch.rand((len(job_ids), 1))
data["company"].x = torch.rand((len(company_ids), 1))

def map_id_to_index(ids):
    return {id_: idx for idx, id_ in enumerate(ids)}

person_map = map_id_to_index(person_ids)
job_map = map_id_to_index(job_ids)
company_map = map_id_to_index(company_ids)

edge_index = {"person": [], "job_posting": [], "company": []}

for e in dataset["edges"]:
    src, tgt, rel = e["source"], e["target"], e["relation"]
    if rel == "applied_to":
        edge_index[rel] = (edge_index.get("applied_to", [[],[]]))
        edge_index["applied_to"][0].append(person_map[src])
        edge_index["applied_to"][1].append(job_map[tgt])
    elif rel == "worked_at":
        edge_index["worked_at"] = (edge_index.get("worked_at", [[],[]]))
        edge_index["worked_at"][0].append(person_map[src])
        edge_index["worked_at"][1].append(company_map[tgt])
    elif rel == "posted_by":
        edge_index["posted_by"] = (edge_index.get("posted_by", [[],[]]))
        edge_index["posted_by"][0].append(job_map[src])
        edge_index["posted_by"][1].append(company_map[tgt])

for key in edge_index:
    src_list, tgt_list = edge_index[key]
    data[key].edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)

# Hetero GNN
class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels=16):
        super().__init__()
        self.conv1 = HeteroConv({
            ('person', 'applied_to', 'job_posting'): GCNConv(-1, hidden_channels),
            ('person', 'worked_at', 'company'): GCNConv(-1, hidden_channels),
            ('job_posting', 'posted_by', 'company'): GCNConv(-1, hidden_channels)
        }, aggr='sum')
        self.lin = torch.nn.Linear(hidden_channels, 1) 

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        person_emb = x_dict["person"]
        job_emb = x_dict["job_posting"]
        scores = torch.sigmoid((person_emb @ job_emb.T))
        return scores

model = HeteroGNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)



y = torch.tensor([random.randint(0,1) for _ in range(len(edge_index["applied_to"][0]))], dtype=torch.float)

for epoch in range(5):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, {k: data[k].edge_index for k in data.edge_types})
    loss = F.binary_cross_entropy(out[edge_index["applied_to"][0], edge_index["applied_to"][1]], y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

print("GNN training complete! Ready for analysis or visualization.")
