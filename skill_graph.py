"""
skill_graph.py
--------------
Area 02 — Potential & Learning Trajectory
Builds a skill adjacency graph and computes:
  - Skill-distance between candidate skills and job requirements
  - Time-to-Productivity estimate per missing skill
  - Adjacency bonus score (candidates who can learn fast rank higher)

Usage:
    graph = build_skill_graph()
    result = compute_learning_trajectory(
        candidate_skills=["Python", "React", "PostgreSQL"],
        required_skills=["Vue", "FastAPI", "Redis"],
        graph=graph,
    )
"""

from dataclasses import dataclass
from typing import Optional

try:
    import networkx as nx
except ImportError:
    raise ImportError("pip install networkx")


# ---------------------------------------------------------------------------
# Skill adjacency definitions
# Distance = how many 'hops' to transfer knowledge
# 1 = very close (same paradigm), 2 = moderate, 3 = harder transfer
# ---------------------------------------------------------------------------

SKILL_EDGES = [
    # Frontend frameworks
    ("React", "Vue", 1), ("React", "Angular", 2), ("Vue", "Angular", 2),
    ("React", "Next.js", 1), ("Vue", "Nuxt.js", 1),
    ("JavaScript", "TypeScript", 1), ("JavaScript", "React", 1),
    ("JavaScript", "Vue", 1), ("JavaScript", "Angular", 1),
    ("TypeScript", "React", 1), ("TypeScript", "Vue", 1),
    ("HTML", "CSS", 1), ("CSS", "Tailwind CSS", 1),

    # Backend frameworks
    ("Python", "FastAPI", 1), ("Python", "Django", 1), ("Python", "Flask", 1),
    ("FastAPI", "Django", 1), ("Flask", "FastAPI", 1), ("Django", "Flask", 1),
    ("Node.js", "Express", 1), ("JavaScript", "Node.js", 1),
    ("Java", "Spring Boot", 1), ("Kotlin", "Spring Boot", 1),
    ("C#", ".NET", 1), ("Ruby", "Rails", 1),

    # Language transfers
    ("Python", "R", 2), ("Java", "Kotlin", 1), ("Java", "Scala", 2),
    ("C++", "Rust", 2), ("C++", "C", 1), ("C", "C++", 1),
    ("Java", "Go", 2), ("Python", "Go", 3), ("JavaScript", "Go", 3),
    ("Python", "Julia", 2), ("Rust", "Go", 2),

    # Databases
    ("PostgreSQL", "MySQL", 1), ("MySQL", "PostgreSQL", 1),
    ("PostgreSQL", "SQLite", 1), ("SQL", "PostgreSQL", 1),
    ("SQL", "MySQL", 1), ("PostgreSQL", "Supabase", 1),
    ("MongoDB", "Redis", 2), ("Redis", "Memcached", 1),
    ("PostgreSQL", "MongoDB", 2), ("MongoDB", "DynamoDB", 2),
    ("PostgreSQL", "DynamoDB", 3),

    # ML / Data
    ("Python", "TensorFlow", 1), ("Python", "PyTorch", 1),
    ("TensorFlow", "PyTorch", 1), ("PyTorch", "TensorFlow", 1),
    ("TensorFlow", "Keras", 1), ("scikit-learn", "PyTorch", 2),
    ("Python", "scikit-learn", 1), ("scikit-learn", "TensorFlow", 2),
    ("Machine Learning", "Deep Learning", 2),
    ("Machine Learning", "NLP", 2), ("Deep Learning", "NLP", 1),
    ("NLP", "Computer Vision", 2),
    ("Python", "Pandas", 1), ("Python", "NumPy", 1),
    ("Pandas", "Spark", 2), ("SQL", "Pandas", 1),

    # DevOps / Cloud
    ("Docker", "Kubernetes", 2), ("Kubernetes", "Helm", 1),
    ("AWS", "GCP", 2), ("AWS", "Azure", 2), ("GCP", "Azure", 2),
    ("Terraform", "AWS", 1), ("Terraform", "GCP", 1),
    ("Docker", "AWS", 2), ("Linux", "Docker", 1),
    ("GitHub Actions", "CI/CD", 1), ("Jenkins", "CI/CD", 1),

    # General
    ("REST APIs", "GraphQL", 2), ("REST APIs", "gRPC", 2),
    ("Microservices", "Docker", 1), ("Microservices", "Kubernetes", 1),
    ("Git", "GitHub Actions", 1),

    # -----------------------------------------------------------------------
    # Domain skill nodes — bridges Github_Analysis output to the graph
    # These match the keys produced by LANGUAGE_SKILL_MAP in Github_Analysis.py
    # -----------------------------------------------------------------------

    # Backend Development
    ("Backend Development", "Python", 1),
    ("Backend Development", "FastAPI", 1),
    ("Backend Development", "Django", 1),
    ("Backend Development", "Flask", 1),
    ("Backend Development", "Node.js", 1),
    ("Backend Development", "Java", 1),
    ("Backend Development", "Go", 2),
    ("Backend Development", "REST APIs", 1),
    ("Backend Development", "Microservices", 2),
    ("Backend Development", "SQL", 1),

    # Frontend Development
    ("Frontend Development", "JavaScript", 1),
    ("Frontend Development", "TypeScript", 1),
    ("Frontend Development", "React", 1),
    ("Frontend Development", "Vue", 1),
    ("Frontend Development", "Angular", 2),
    ("Frontend Development", "HTML", 1),
    ("Frontend Development", "CSS", 1),

    # Machine Learning / AI
    ("Machine Learning", "Python", 1),
    ("Machine Learning", "TensorFlow", 1),
    ("Machine Learning", "PyTorch", 1),
    ("Machine Learning", "scikit-learn", 1),
    ("Machine Learning", "Pandas", 1),
    ("Machine Learning", "NumPy", 1),
    ("AI/ML", "Python", 1),
    ("AI/ML", "TensorFlow", 1),
    ("AI/ML", "PyTorch", 1),
    ("AI/ML", "Machine Learning", 1),
    ("AI/ML", "Deep Learning", 1),
    ("Data Science", "Python", 1),
    ("Data Science", "Machine Learning", 1),
    ("Data Science", "Pandas", 1),
    ("Data Science", "SQL", 1),
    ("Data Science", "R", 2),
    ("Data Analysis", "SQL", 1),
    ("Data Analysis", "Python", 1),
    ("Data Analysis", "Pandas", 1),

    # DevOps / Cloud / Infrastructure
    ("DevOps", "Docker", 1),
    ("DevOps", "Kubernetes", 2),
    ("DevOps", "Linux", 1),
    ("DevOps", "CI/CD", 1),
    ("DevOps", "GitHub Actions", 1),
    ("DevOps", "Terraform", 2),
    ("Cloud Infrastructure", "AWS", 1),
    ("Cloud Infrastructure", "GCP", 1),
    ("Cloud Infrastructure", "Azure", 1),
    ("Cloud Infrastructure", "Docker", 1),
    ("Cloud Infrastructure", "Terraform", 1),
    ("Scripting", "Python", 1),
    ("Scripting", "Linux", 1),
    ("Containerization", "Docker", 1),
    ("Containerization", "Kubernetes", 2),

    # Systems / Low-level
    ("Systems Programming", "C++", 1),
    ("Systems Programming", "C", 1),
    ("Systems Programming", "Rust", 1),
    ("High-Performance Computing", "C++", 1),
    ("High-Performance Computing", "Rust", 2),
    ("Embedded Systems", "C", 1),
    ("Embedded Systems", "C++", 1),
    ("Operating Systems", "Linux", 1),
    ("Operating Systems", "C", 1),
    ("Kernel Development", "C", 1),
    ("Kernel Development", "Linux", 1),

    # Database
    ("Database Design", "SQL", 1),
    ("Database Design", "PostgreSQL", 1),
    ("Database Design", "MongoDB", 2),
    ("Relational Databases", "SQL", 1),
    ("Relational Databases", "PostgreSQL", 1),
    ("Relational Databases", "MySQL", 1),

    # Mobile
    ("Android Development", "Kotlin", 1),
    ("Android Development", "Java", 1),
    ("iOS Development", "Swift", 1),
    ("Mobile Development", "Dart", 1),
    ("Mobile Development", "Kotlin", 2),
    ("Cross-Platform Development", "Dart", 1),

    # Web / General
    ("Web Development", "JavaScript", 1),
    ("Web Development", "HTML", 1),
    ("Web Development", "CSS", 1),
    ("Web Development", "React", 1),
    ("Web Design", "HTML", 1),
    ("Web Design", "CSS", 1),
    ("UI/UX", "HTML", 1),
    ("UI/UX", "CSS", 1),
    ("UI/UX", "React", 2),

    # Enterprise / Architecture
    ("Enterprise Architecture", "Java", 1),
    ("Enterprise Architecture", "Spring Boot", 1),
    ("Enterprise Architecture", "Microservices", 2),
    ("System Architecture", "Microservices", 1),
    ("System Architecture", "Docker", 2),
    ("System Architecture", "REST APIs", 1),

    # Data Engineering
    ("Data Engineering", "Python", 1),
    ("Data Engineering", "Spark", 1),
    ("Data Engineering", "SQL", 1),
    ("Big Data", "Spark", 1),
    ("Big Data", "Python", 2),

    # Statistical / Research
    ("Statistical Modeling", "R", 1),
    ("Statistical Modeling", "Python", 2),

    # Game / Scripting
    ("Game Development", "C++", 1),
    ("Game Development", "C#", 1),
    ("Game Development", "Lua", 2),

    # Memory / Security
    ("Memory Safety", "Rust", 1),
    ("Security", "Python", 2),
    ("Cryptography", "Python", 2),

    # .NET
    (".NET", "C#", 1),
    (".NET", "Java", 2),

    # WebAssembly
    ("WebAssembly", "Rust", 1),
    ("WebAssembly", "C++", 2),

    # Algorithms / CS fundamentals
    ("Algorithms", "Python", 1),
    ("Algorithms", "C++", 1),
    ("Algorithms", "Java", 1),
    ("Competitive Programming", "C++", 1),
    ("Competitive Programming", "Algorithms", 1),
]

# Approximate weeks to learn a skill given adjacency distance
WEEKS_PER_HOP = {1: 2, 2: 6, 3: 14}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_skill_graph() -> nx.Graph:
    """Build an undirected weighted skill adjacency graph."""
    G = nx.Graph()
    for source, target, distance in SKILL_EDGES:
        G.add_node(source)
        G.add_node(target)
        G.add_edge(source, target, distance=distance, weight=1.0 / distance)
    return G


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SkillGap:
    skill: str
    distance: int                    # graph hops from closest candidate skill
    via: str                         # bridging skill in candidate's profile
    weeks_to_acquire: int
    confidence: str                  # "high" / "medium" / "low"


@dataclass
class LearningTrajectory:
    candidate_skills: list[str]
    required_skills: list[str]
    matched_skills: list[str]
    skill_gaps: list[SkillGap]
    adjacency_score: float           # 0-100 bonus for close adjacencies
    total_weeks_to_productivity: int
    productivity_band: str           # "immediate" / "fast" / "moderate" / "slow"
    reasoning: str


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_learning_trajectory(
    candidate_skills: list[str],
    required_skills: list[str],
    graph: nx.Graph,
) -> LearningTrajectory:
    """
    For each required skill not in the candidate's profile,
    compute shortest graph distance from any skill they DO have.
    """
    candidate_set = {s.strip() for s in candidate_skills}
    required_set  = {s.strip() for s in required_skills}

    matched = [s for s in required_set if s in candidate_set]
    missing = [s for s in required_set if s not in candidate_set]

    skill_gaps: list[SkillGap] = []
    total_weeks = 0

    for req_skill in missing:
        best_distance = 999
        best_via = "no path"

        # Add req_skill temporarily if it's not in the graph
        # so NodeNotFound doesn't silently skip it
        added = req_skill not in graph
        if added:
            graph.add_node(req_skill)

        for cand_skill in candidate_set:
            if cand_skill not in graph:
                continue
            try:
                path_length = nx.shortest_path_length(
                    graph, cand_skill, req_skill, weight=None
                )
                if path_length < best_distance:
                    best_distance = path_length
                    best_via = cand_skill
            except nx.NetworkXNoPath:
                continue
            except nx.NodeNotFound:
                continue

        if added:
            graph.remove_node(req_skill)

        if best_distance == 999:
            weeks = 20
            confidence = "low"
        else:
            weeks = WEEKS_PER_HOP.get(best_distance, 20)
            confidence = {1: "high", 2: "medium"}.get(best_distance, "low")

        skill_gaps.append(SkillGap(
            skill=req_skill,
            distance=best_distance if best_distance < 999 else -1,
            via=best_via,
            weeks_to_acquire=weeks,
            confidence=confidence,
        ))
        total_weeks = max(total_weeks, weeks)

    # Adjacency bonus score
    reachable = sum(1 for g in skill_gaps if g.distance in (1, 2))
    adjacency_score = round(
        (len(matched) / len(required_set) * 60)
        + (reachable / max(len(missing), 1) * 40),
        1
    ) if required_set else 0.0

    # Productivity band
    if total_weeks == 0:
        band = "immediate"
    elif total_weeks <= 4:
        band = "fast"
    elif total_weeks <= 10:
        band = "moderate"
    else:
        band = "slow"

    # Reasoning narrative
    gap_summaries = []
    for g in skill_gaps:
        if g.distance == -1:
            gap_summaries.append(f"{g.skill} (no adjacent skills — cold start ~{g.weeks_to_acquire}w)")
        else:
            gap_summaries.append(f"{g.skill} ({g.distance}-hop from {g.via}, ~{g.weeks_to_acquire}w)")

    reasoning = (
        f"Candidate directly matches {len(matched)}/{len(required_set)} required skills. "
    )
    if gap_summaries:
        reasoning += f"Gaps: {'; '.join(gap_summaries)}. "
    reasoning += (
        f"Estimated time-to-productivity: {total_weeks} weeks ({band}). "
        f"Adjacency score: {adjacency_score}/100."
    )

    return LearningTrajectory(
        candidate_skills=list(candidate_set),
        required_skills=list(required_set),
        matched_skills=matched,
        skill_gaps=skill_gaps,
        adjacency_score=adjacency_score,
        total_weeks_to_productivity=total_weeks,
        productivity_band=band,
        reasoning=reasoning,
    )


if __name__ == "__main__":
    graph = build_skill_graph()
    result = compute_learning_trajectory(
        candidate_skills=["Backend Development", "Machine Learning", "DevOps"],
        required_skills=["FastAPI", "PyTorch", "Kubernetes", "React"],
        graph=graph,
    )
    print(result.reasoning)
    for gap in result.skill_gaps:
        print(f"  → {gap.skill}: {gap.distance} hops via {gap.via} (~{gap.weeks_to_acquire}w, {gap.confidence} confidence)")
