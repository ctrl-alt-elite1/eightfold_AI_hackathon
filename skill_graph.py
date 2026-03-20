import networkx as nx
import re
import torch
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------------------------------------
# Distance 1: Direct evolution (e.g., JS -> TS, React -> Next.js)
# Distance 2: Shared paradigm (e.g., SQL -> NoSQL, C++ -> Rust)
# Distance 3: Major pivot (e.g., Python -> Go, Java -> C)
# ---------------------------------------------------------------------------

SKILL_EDGES = [
    # --- WEB ECOSYSTEM ---
    ("JavaScript", "TypeScript", 1), ("JavaScript", "Node.js", 1),
    ("TypeScript", "React", 1), ("TypeScript", "Vue", 1), ("TypeScript", "Angular", 1),
    ("React", "Next.js", 1), ("React", "Redux", 1), ("React", "Tailwind CSS", 1),
    ("Vue", "Nuxt.js", 1), ("Vue", "Pinia", 1),
    ("HTML", "CSS", 1), ("CSS", "Sass", 1), ("CSS", "Tailwind CSS", 1),
    ("Node.js", "Express", 1), ("Node.js", "Fastify", 1), ("Node.js", "NestJS", 1),

    # --- SYSTEMS & LOW-LEVEL ---
    ("C", "C++", 1), ("C++", "Rust", 2), ("C", "Assembly", 2),
    ("C++", "CUDA", 2), ("C++", "Embedded Systems", 1),
    ("C", "Kernel Development", 1), ("C", "Operating Systems", 1),
    ("Linux", "Kernel Development", 1), ("Linux", "Shell Scripting", 1),
    ("Systems Programming", "Rust", 1), ("Systems Programming", "C++", 1),
    ("Memory Management", "C", 1), ("Memory Management", "C++", 1),
    ("Low-Level Control", "C", 1), ("Hardware Interaction", "Embedded Systems", 1),

    # --- BACKEND & ARCHITECTURE ---
    ("Python", "FastAPI", 1), ("Python", "Django", 1), ("Python", "Flask", 1),
    ("Java", "Spring Boot", 1), ("Java", "Kotlin", 1), ("Java", "Scala", 2),
    ("C#", ".NET", 1), ("C#", "Unity", 2),
    ("Go", "Microservices", 1), ("Go", "Cloud Native", 1),
    ("REST APIs", "GraphQL", 2), ("REST APIs", "gRPC", 2),
    ("Microservices", "Docker", 1), ("Microservices", "Kubernetes", 1),
    ("Distributed Systems", "Go", 2), ("Distributed Systems", "Scala", 2),

    # --- DATA SCIENCE & AI ---
    ("Python", "Pandas", 1), ("Python", "NumPy", 1), ("Python", "scikit-learn", 1),
    ("Python", "PyTorch", 1), ("Python", "TensorFlow", 1),
    ("PyTorch", "Deep Learning", 1), ("TensorFlow", "Deep Learning", 1),
    ("Deep Learning", "NLP", 1), ("Deep Learning", "Computer Vision", 1),
    ("Machine Learning", "Data Science", 1), ("Data Science", "R", 2),
    ("Big Data", "Spark", 1), ("Spark", "Hadoop", 2), ("Data Engineering", "SQL", 1),
    ("Data Engineering", "Airflow", 1),

    # --- DATABASES ---
    ("SQL", "PostgreSQL", 1), ("SQL", "MySQL", 1), ("SQL", "SQL Server", 1),
    ("PostgreSQL", "MySQL", 1), ("PostgreSQL", "NoSQL", 2),
    ("NoSQL", "MongoDB", 1), ("NoSQL", "Redis", 1), ("NoSQL", "Cassandra", 2),
    ("Redis", "Memcached", 1), ("Database Design", "SQL", 1),

    # --- CLOUD & DEVOPS ---
    ("Docker", "Containerization", 1), ("Docker", "Kubernetes", 2),
    ("Kubernetes", "Helm", 1), ("Kubernetes", "Service Mesh", 2),
    ("AWS", "Cloud Infrastructure", 1), ("GCP", "Cloud Infrastructure", 1), ("Azure", "Cloud Infrastructure", 1),
    ("AWS", "Terraform", 1), ("Cloud Infrastructure", "Terraform", 1),
    ("CI/CD", "GitHub Actions", 1), ("CI/CD", "Jenkins", 1), ("CI/CD", "GitLab CI", 1),
    ("Automation", "Ansible", 1), ("Automation", "Shell Scripting", 1),

    # --- MOBILE ---
    ("Kotlin", "Android Development", 1), ("Swift", "iOS Development", 1),
    ("Dart", "Flutter", 1), ("Flutter", "Mobile Development", 1),
    ("React Native", "React", 1),

    # --- DOMAIN BRIDGES (Cross-matching triggers) ---
    ("Backend Development", "REST APIs", 1), ("Frontend Development", "UI/UX", 2),
    ("Security", "Cryptography", 2), ("Security", "Network Security", 1),
    ("Performance Tuning", "High-Performance Computing", 1),
    ("Software Architecture", "System Design", 1),
]

WEEKS_PER_HOP = {1: 2, 2: 6, 3: 14}

class SemanticSkillMapper:
    def __init__(self, graph: nx.Graph):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.graph_nodes = list(graph.nodes())
        self.node_embeddings = self.model.encode(self.graph_nodes, convert_to_tensor=True)

    def snap_to_graph(self, raw_skills: list[str]) -> dict[str, str]:
        mapping = {}
        if not raw_skills: return mapping
        cleaned = [re.sub(r'\s*\((Core|Secondary)\)', '', s, flags=re.IGNORECASE).strip() for s in raw_skills]
        input_embeddings = self.model.encode(cleaned, convert_to_tensor=True)
        cosine_scores = util.cos_sim(input_embeddings, self.node_embeddings)

        for i, original in enumerate(raw_skills):
            best_idx = torch.argmax(cosine_scores[i]).item()
            if cosine_scores[i][best_idx].item() >= 0.55:
                mapping[original] = self.graph_nodes[best_idx]
            else:
                mapping[original] = cleaned[i]
        return mapping

@dataclass
class SkillGap:
    original_skill: str
    mapped_node: str
    distance: int
    via: str
    weeks: int
    confidence: str

@dataclass
class LearningTrajectory:
    matched: list[str]
    gaps: list[SkillGap]
    score: float
    weeks: int
    band: str
    reasoning: str

def build_skill_graph():
    G = nx.Graph()
    for s, t, d in SKILL_EDGES:
        G.add_edge(s, t, distance=d)
    return G

def compute_learning_trajectory(candidate_skills, required_skills, graph, mapper):
    mapped_cands = mapper.snap_to_graph(candidate_skills)
    mapped_reqs = mapper.snap_to_graph(required_skills)
    cand_nodes = set(mapped_cands.values())
    
    matched_orig = []
    gaps = []
    max_weeks = 0

    for raw_req, m_req in mapped_reqs.items():
        if m_req in cand_nodes:
            matched_orig.append(raw_req)
            continue

        best_d, best_v = 999, "None"
        added = m_req not in graph
        if added: graph.add_node(m_req)

        for m_cand in cand_nodes:
            if m_cand in graph:
                try:
                    d = nx.shortest_path_length(graph, m_cand, m_req, weight="distance")
                    if d < best_d: best_d, best_v = d, m_cand
                except: continue

        if added: graph.remove_node(m_req)
        
        weeks = WEEKS_PER_HOP.get(best_d, int(best_d * 4)) if best_d < 999 else 20
        conf = "high" if best_d == 1 else "medium" if best_d <= 3 else "low"
        
        gaps.append(SkillGap(raw_req, m_req, best_d if best_d < 999 else -1, best_v, weeks, conf))
        max_weeks = max(max_weeks, weeks)

    adj_score = round((len(matched_orig)/len(required_skills)*60) + (sum(1 for g in gaps if 0 < g.distance <= 3)/max(len(gaps),1)*40), 1) if required_skills else 0
    band = "immediate" if max_weeks == 0 else "fast" if max_weeks <= 4 else "moderate" if max_weeks <= 10 else "slow"
    
    reasoning = f"Directly matches {len(matched_orig)}/{len(required_skills)} skills. "
    reasoning += f"Trajectory: {max_weeks}w ({band}). Score: {adj_score}/100."

    return LearningTrajectory(matched_orig, gaps, adj_score, max_weeks, band, reasoning)