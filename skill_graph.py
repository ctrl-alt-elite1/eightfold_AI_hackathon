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
        # Add both nodes if not present
        G.add_node(source)
        G.add_node(target)
        # Use inverse distance as weight so shorter path = stronger connection
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
    # Normalise case for matching
    candidate_set = {s.strip() for s in candidate_skills}
    required_set = {s.strip() for s in required_skills}

    matched = [s for s in required_set if s in candidate_set]
    missing = [s for s in required_set if s not in candidate_set]

    skill_gaps: list[SkillGap] = []
    total_weeks = 0

    for req_skill in missing:
        best_distance = 999
        best_via = "no path"

        for cand_skill in candidate_set:
            if cand_skill not in graph or req_skill not in graph:
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

        if best_distance == 999:
            # No graph path — treat as cold start
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
        total_weeks = max(total_weeks, weeks)  # parallel learning, use max

    # Adjacency bonus score: proportion of missing skills reachable within 2 hops
    reachable = sum(1 for g in skill_gaps if g.distance in (1, 2))
    adjacency_score = round(
        (len(matched) / len(required_set) * 60)        # 60 pts for direct matches
        + (reachable / max(len(missing), 1) * 40),     # 40 pts for near adjacencies
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
        candidate_skills=["React", "JavaScript", "Python", "PostgreSQL"],
        required_skills=["Vue", "FastAPI", "Redis", "Kubernetes"],
        graph=graph,
    )
    print(result.reasoning)
    for gap in result.skill_gaps:
        print(f"  → {gap.skill}: {gap.distance} hops via {gap.via} (~{gap.weeks_to_acquire}w, {gap.confidence} confidence)")
