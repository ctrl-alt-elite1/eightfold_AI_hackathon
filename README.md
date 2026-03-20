# Talent Intelligence System
**Techkriti '26 × Eightfold AI Hackathon — Impact Areas 01 & 02**

---

## The Problem

Modern hiring is broken in two directions simultaneously.

Recruiters are drowning in AI-optimised resumes that are largely synthetic — indistinguishable from genuine signals of competence. At the same time, static credential matching misses candidates who don't yet have a required skill but can acquire it faster than anyone else in the pool.

The result: organisations hire the best resume, not the best engineer.

---

## Our Solution

A **GitHub-native talent intelligence system** that bypasses the resume entirely and evaluates candidates on what they have actually built.

Instead of parsing PDFs, we pull verified signals directly from a candidate's public GitHub profile — commit volume, language depth, project complexity, consistency, and learning trajectory — and map them semantically against a job description using vector embeddings.

**The core insight:** a candidate who doesn't have a required skill today but has a 1-hop adjacency in their skill graph will outperform a keyword match who hasn't touched the technology in two years.

---

## Features

| Feature | What it does |
|---|---|
| **GitHub Signal Extraction** | Parallel verification of commit depth, language distribution, project complexity, and recency across public repos |
| **Developer Archetype** | Classifies candidates as Maintainer / Builder / Explorer / Generalist based on activity patterns |
| **Learning Velocity** | Detects new languages adopted in the last 12 months to quantify adaptability |
| **Semantic JD Matching** | `all-MiniLM-L6-v2` embeddings map JD requirements to candidate skills with per-skill confidence scores |
| **Weighted Scoring** | Core skills (80%) vs Secondary skills (20%) — role-critical gaps penalise more than peripheral ones |
| **Skill Adjacency Graph** | NetworkX graph computes shortest path from candidate's existing skills to any gap, producing a Time-to-Productivity estimate per missing skill |
| **Hire Recommendation** | STRONG HIRE / INTERVIEW / NEEDS REVIEW / PASS with explicit reasoning |
| **Candidate Comparison** | Side-by-side head-to-head table with composite winner declaration |
| **Bias Check** | Re-scores after stripping gendered pronouns from JD — delta ≤ 3 pts = pass |

---

## Architecture

```
GitHub Profile
      │
      ▼
Github_Analysis.py  ──────────────────────────────────────────┐
  Layer 1: Repo fetch + cache (1hr TTL)                        │
  Layer 2: Parallel commit verification (ThreadPoolExecutor)   │
  Layer 3: Skill evidence scoring (commit-weighted confidence) │
  Layer 4: Developer archetype + learning velocity             │
  Layer 5: Engineering intelligence score                      │
  Layer 6: Analytics (tech evolution timeline)                 │
      │                                                        │
      ▼                                                        ▼
Job_Matcher.py                                          skill_graph.py
  JD requirement extraction                            NetworkX adjacency graph
  (regex taxonomy + optional Ollama)                  Skill-distance BFS
  Sentence-transformer embeddings                     Time-to-Productivity estimate
  Cosine similarity matrix                            Adjacency bonus score
  Core (80%) + Secondary (20%) weighted score
      │                                                        │
      └──────────────────────┬─────────────────────────────────┘
                             ▼
                          app.py
                    Streamlit UI
                    Hire recommendation engine
                    Comparison mode
                    Bias check
                    Explainability report
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Optional: set GitHub token (avoids 60 req/hr rate limit)
export GITHUB_TOKEN=your_token_here

# 3. Run
streamlit run app.py
```

---

## File Structure

```
app.py               — Streamlit UI + hire recommendation + comparison mode
Github_Analysis.py   — GitHub signal extraction (6-layer pipeline)
Job_Matcher.py       — Semantic JD matching via sentence-transformers
skill_graph.py       — Skill adjacency graph + Time-to-Productivity engine
requirements.txt     — Python dependencies
```

---

## Key Design Decisions

### Why GitHub instead of resumes?
Resumes are self-reported and increasingly AI-generated. GitHub commit history is a verified, timestamped record of actual work. It's much harder to fake 200 commits across 15 repos than it is to add "Machine Learning" to a PDF.

### Why sentence-transformers over keyword matching?
Keyword matching fails on synonyms and adjacent skills. A candidate with "PyTorch" experience is highly relevant to a "TensorFlow" requirement — cosine similarity catches this, keyword scanning misses it entirely.

### Why the skill graph?
A candidate who knows React and needs to learn Vue has a 2-week ramp-up. A candidate who knows Java and needs to learn Rust has a 14-week ramp-up. These are not equivalent "missing skills" — treating them as such produces misleading match scores. The graph makes the difference explicit and quantified.

### The 80/20 weighted scoring
Not all JD requirements are equal. A systems role that lists "C++" as a core requirement and "Agile" as secondary should penalise a C++ gap far more than an Agile gap. The Core/Secondary weighting reflects this.

---

## Demo Flow

1. Paste any job description into the right panel
2. Enter a GitHub username (or two for comparison mode)
3. Hit **Analyse Candidate**
4. Walk through:
   - **Hire recommendation** with explicit reasoning at the top
   - **GitHub signals** — consistency, depth, exploration, domain skill bars
   - **Semantic matches** — each requirement mapped to specific GitHub evidence with confidence %
   - **Learning trajectory** — Time-to-Productivity per gap with skill-path reasoning
   - **Bias check** — score delta with/without demographic language
   - **Comparison table** — head-to-head metrics with composite winner (compare mode)

**Demo tip:** Use two GitHub profiles where one has fewer direct matches but higher learning velocity — show that the system ranks the fast learner higher than a stale keyword match. That's the core innovation.

---

## Known Limitations

- Skill evidence is domain-level (e.g. "Backend Development") rather than tool-specific (e.g. "FastAPI") — improving this requires README parsing which adds API call overhead
- Private repositories are not accessible — public activity may underrepresent some candidates
- Time-to-Productivity estimates are heuristic (weeks-per-hop table); a production system would calibrate these on outcome data
- Ollama/phi3 for JD extraction is optional — regex taxonomy fallback is used when unavailable

---

## Built by

ctrl-alt-elite · Techkriti '26
