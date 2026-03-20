"""
Github_Analysis.py
GitHub Engineering Intelligence Module — AI Hiring Intelligence MVP
"""

import os
import json
import time
import requests
from datetime import datetime, timezone, timedelta
from typing import Any


# --- API ---
GITHUB_API_BASE  = "https://api.github.com"
MAX_REPOS_FETCH  = 100

# --- CACHE ---
CACHE_DIR        = ".github_cache"
CACHE_TTL        = 3600          # seconds

# --- ACTIVITY ---
ACTIVE_REPO_DAYS = 180

# --- ARCHETYPE THRESHOLDS ---
MAINTAINER_CONSISTENCY_MIN = 70
BUILDER_DEPTH_MIN          = 65
BUILDER_REPOS_MIN          = 3
EXPLORER_REPOS_MIN         = 8
EXPLORER_EXPLORATION_MIN   = 60

# --- LEARNING VELOCITY ---
VELOCITY_HIGH   = 2
VELOCITY_MEDIUM = 1

# --- CONFIDENCE ---
CONFIDENCE_HIGH_MIN   = 8
CONFIDENCE_MEDIUM_MIN = 3

# --- SCORING WEIGHTS ---
WEIGHT_SKILLS    = 0.35
WEIGHT_BEHAVIOR  = 0.25
WEIGHT_DEPTH     = 0.15
WEIGHT_POTENTIAL = 0.15
WEIGHT_ACTIVITY  = 0.10

# --- BEHAVIOR WEIGHTS ---
BEHAVIOR_CONSISTENCY = 0.4
BEHAVIOR_EXPLORATION = 0.3
BEHAVIOR_DEPTH       = 0.3

# --- POTENTIAL WEIGHTS ---
POTENTIAL_DEPTH       = 0.4
POTENTIAL_CONSISTENCY = 0.3
POTENTIAL_EXPLORATION = 0.3

# --- STRENGTH THRESHOLDS ---
STRENGTH_SKILL_MIN       = 75
STRENGTH_CONSISTENCY_MIN = 70
STRENGTH_DEPTH_MIN       = 65

# --- RISK THRESHOLDS ---
RISK_DEPTH_MAX       = 50
RISK_CONSISTENCY_MAX = 40
RISK_REPOS_MIN       = 3

# --- LANGUAGE → SKILL MAP ---
LANGUAGE_SKILL_MAP: dict[str, list[str]] = {
    "Python":           ["Backend Development", "AI/ML", "Scripting"],
    "Jupyter Notebook": ["AI/ML", "Data Science"],
    "R":                ["Data Science", "AI/ML"],
    "JavaScript":       ["Frontend Development", "Backend Development"],
    "TypeScript":       ["Frontend Development", "Backend Development"],
    "HTML":             ["Frontend Development"],
    "CSS":              ["Frontend Development"],
    "Java":             ["Backend Development", "Android"],
    "Kotlin":           ["Android", "Backend Development"],
    "Swift":            ["iOS Development"],
    "C":                ["Systems Programming"],
    "C++":              ["Systems Programming", "Competitive Programming"],
    "C#":               ["Backend Development", ".NET"],
    "Go":               ["Backend Development", "DevOps"],
    "Rust":             ["Systems Programming"],
    "PHP":              ["Backend Development", "Web Development"],
    "Ruby":             ["Backend Development", "Web Development"],
    "Shell":            ["DevOps", "Scripting"],
    "Dockerfile":       ["DevOps"],
    "YAML":             ["DevOps"],
    "SQL":              ["Database"],
    "PLpgSQL":          ["Database"],
    "Scala":            ["Backend Development", "Data Engineering"],
    "Dart":             ["Mobile Development"],
    "Lua":              ["Scripting", "Game Development"],
}


# ---------------------------------------------------------------------------
# CACHE UTILITIES
# ---------------------------------------------------------------------------

def _cache_path(username: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{username.lower()}.json")


def _load_cache(username: str) -> list | None:
    path = _cache_path(username)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if time.time() - cached.get("cached_at", 0) < CACHE_TTL:
            data = cached.get("data")
            if isinstance(data, list):
                return data
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _save_cache(username: str, data: list) -> None:
    try:
        with open(_cache_path(username), "w", encoding="utf-8") as f:
            json.dump({"cached_at": time.time(), "data": data}, f)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# LAYER 1 — DATA COLLECTION
# ---------------------------------------------------------------------------

def fetch_repositories(username: str) -> list[dict[str, Any]]:
    if not username or not isinstance(username, str):
        return []

    username = username.strip()
    cached = _load_cache(username)
    if cached is not None:
        return cached

    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        repos_resp = requests.get(
            f"{GITHUB_API_BASE}/users/{username}/repos",
            headers=headers,
            params={"per_page": MAX_REPOS_FETCH, "sort": "updated", "type": "owner"},
            timeout=10,
        )
        if repos_resp.status_code != 200:
            return []

        raw = repos_resp.json()
        if not isinstance(raw, list):
            return []

        _save_cache(username, raw)
        return raw

    except (requests.RequestException, ValueError):
        return []


def normalize_repo_data(raw_repos: list[dict]) -> list[dict[str, Any]]:
    if not raw_repos or not isinstance(raw_repos, list):
        return []

    normalized = []
    for repo in raw_repos:
        if not isinstance(repo, dict):
            continue
        if repo.get("fork", False):
            continue

        normalized.append({
            "name":        str(repo.get("name") or ""),
            "language":    repo.get("language") or None,
            "description": str(repo.get("description") or "").strip(),
            "stars":       int(repo.get("stargazers_count") or 0),
            "forks":       int(repo.get("forks_count") or 0),
            "created_at":  _parse_date(repo.get("created_at")),
            "updated_at":  _parse_date(repo.get("updated_at")),
            "topics":      list(repo.get("topics") or []),
        })

    return normalized


def _parse_date(date_str: Any) -> datetime | None:
    if not date_str or not isinstance(date_str, str):
        return None
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# LAYER 2 — SKILL EVIDENCE ENGINE
# ---------------------------------------------------------------------------

def analyze_skill_evidence(repos: list[dict[str, Any]]) -> dict[str, Any]:
    if not repos:
        return {"skills": {}, "skill_score": 0}

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=ACTIVE_REPO_DAYS)
    total = len(repos)

    # Count per language: total repos, recent repos
    lang_total: dict[str, int] = {}
    lang_recent: dict[str, int] = {}

    for repo in repos:
        lang = repo.get("language")
        if not lang:
            continue
        lang_total[lang] = lang_total.get(lang, 0) + 1
        updated = repo.get("updated_at")
        if updated and updated >= cutoff:
            lang_recent[lang] = lang_recent.get(lang, 0) + 1

    if not lang_total:
        return {"skills": {}, "skill_score": 0}

    primary_lang = max(lang_total, key=lambda l: lang_total[l])

    # Build skill confidence per language
    lang_confidence: dict[str, float] = {}
    for lang, count in lang_total.items():
        freq_score    = min(count / total, 1.0) * 50
        recency_score = min((lang_recent.get(lang, 0) / count), 1.0) * 30
        primary_bonus = 20 if lang == primary_lang else 0
        lang_confidence[lang] = round(freq_score + recency_score + primary_bonus, 1)

    # Map languages to skills
    skill_scores: dict[str, list[float]] = {}
    for lang, confidence in lang_confidence.items():
        for skill in LANGUAGE_SKILL_MAP.get(lang, [lang]):
            skill_scores.setdefault(skill, []).append(confidence)

    skills = {
        skill: round(sum(scores) / len(scores), 1)
        for skill, scores in skill_scores.items()
    }

    skill_score = round(sum(skills.values()) / len(skills), 1) if skills else 0

    return {"skills": skills, "skill_score": skill_score}


# ---------------------------------------------------------------------------
# LAYER 3 — ENGINEERING BEHAVIOR ANALYSIS
# ---------------------------------------------------------------------------

def analyze_engineering_behavior(repos: list[dict[str, Any]]) -> dict[str, Any]:
    empty = {
        "consistency": 0,
        "exploration": 0,
        "depth": 0,
        "behavior_score": 0,
    }

    if not repos:
        return empty

    now    = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=ACTIVE_REPO_DAYS)
    total  = len(repos)

    active_count = sum(
        1 for r in repos
        if r.get("updated_at") and r["updated_at"] >= cutoff
    )

    unique_langs = len(set(
        r["language"] for r in repos if r.get("language")
    ))

    repos_with_desc = sum(1 for r in repos if r.get("description"))

    consistency = round((active_count / total) * 100, 1)
    exploration = round(min(unique_langs * 15, 100), 1)
    depth       = round((repos_with_desc / total) * 100, 1)

    behavior_score = round(
        consistency * BEHAVIOR_CONSISTENCY +
        exploration * BEHAVIOR_EXPLORATION +
        depth       * BEHAVIOR_DEPTH,
        1
    )

    return {
        "consistency":    consistency,
        "exploration":    exploration,
        "depth":          depth,
        "behavior_score": behavior_score,
    }


# ---------------------------------------------------------------------------
# LAYER 4 — DEVELOPER PROFILE ENGINE
# ---------------------------------------------------------------------------

def classify_developer_profile(
    repos: list[dict[str, Any]],
    behavior: dict[str, Any],
    skills: dict[str, Any],
) -> dict[str, Any]:
    empty = {
        "archetype":         "Insufficient Data",
        "learning_velocity": "LOW",
        "potential_score":   0,
    }

    if not repos:
        return empty

    total       = len(repos)
    consistency = behavior.get("consistency", 0)
    exploration = behavior.get("exploration", 0)
    depth       = behavior.get("depth", 0)

    # Archetype — priority: Maintainer > Builder > Explorer > Generalist
    if consistency >= MAINTAINER_CONSISTENCY_MIN:
        archetype = "Maintainer"
    elif depth >= BUILDER_DEPTH_MIN and total >= BUILDER_REPOS_MIN:
        archetype = "Builder"
    elif total >= EXPLORER_REPOS_MIN and exploration >= EXPLORER_EXPLORATION_MIN:
        archetype = "Explorer"
    else:
        archetype = "Generalist"

    # Learning velocity — new languages in last 12 months
    now          = datetime.now(timezone.utc)
    cutoff_12m   = now - timedelta(days=365)

    older_langs = set(
        r["language"] for r in repos
        if r.get("language") and r.get("created_at")
        and r["created_at"] < cutoff_12m
    )
    recent_langs = set(
        r["language"] for r in repos
        if r.get("language") and r.get("created_at")
        and r["created_at"] >= cutoff_12m
    )
    new_lang_count = len(recent_langs - older_langs)

    if not older_langs:
        learning_velocity = "LOW"
    elif new_lang_count >= VELOCITY_HIGH:
        learning_velocity = "HIGH"
    elif new_lang_count >= VELOCITY_MEDIUM:
        learning_velocity = "MEDIUM"
    else:
        learning_velocity = "LOW"

    potential_score = round(
        depth       * POTENTIAL_DEPTH +
        consistency * POTENTIAL_CONSISTENCY +
        exploration * POTENTIAL_EXPLORATION,
        1
    )

    return {
        "archetype":         archetype,
        "learning_velocity": learning_velocity,
        "potential_score":   potential_score,
    }


# ---------------------------------------------------------------------------
# LAYER 5 — ENGINEERING INTELLIGENCE SCORING
# ---------------------------------------------------------------------------

def compute_engineering_scores(
    skills: dict[str, Any],
    behavior: dict[str, Any],
    profile: dict[str, Any],
    repo_count: int,
) -> dict[str, Any]:
    skill_score     = skills.get("skill_score", 0)
    behavior_score  = behavior.get("behavior_score", 0)
    depth           = behavior.get("depth", 0)
    potential_score = profile.get("potential_score", 0)
    consistency     = behavior.get("consistency", 0)
    exploration     = behavior.get("exploration", 0)
    activity_score  = min(repo_count * 10, 100)

    overall_score = round(max(0, min(100,
        skill_score     * WEIGHT_SKILLS    +
        behavior_score  * WEIGHT_BEHAVIOR  +
        depth           * WEIGHT_DEPTH     +
        potential_score * WEIGHT_POTENTIAL +
        activity_score  * WEIGHT_ACTIVITY
    )), 1)

    breakdown = {
        "skills":    round(skill_score    * WEIGHT_SKILLS,    1),
        "behavior":  round(behavior_score * WEIGHT_BEHAVIOR,  1),
        "depth":     round(depth          * WEIGHT_DEPTH,     1),
        "potential": round(potential_score* WEIGHT_POTENTIAL, 1),
        "activity":  round(activity_score * WEIGHT_ACTIVITY,  1),
    }

    strengths = []
    if skill_score     > STRENGTH_SKILL_MIN:       strengths.append("Strong technical skill evidence")
    if consistency     > STRENGTH_CONSISTENCY_MIN:  strengths.append("Consistent development activity")
    if depth           > STRENGTH_DEPTH_MIN:        strengths.append("Strong project completion signals")

    risks = []
    if depth       < RISK_DEPTH_MAX:        risks.append("Limited project depth")
    if consistency < RISK_CONSISTENCY_MAX:  risks.append("Low recent activity")

    return {
        "overall_score": overall_score,
        "breakdown":     breakdown,
        "strengths":     strengths,
        "risks":         risks,
    }


# ---------------------------------------------------------------------------
# LAYER 6 — ANALYTICS DATA GENERATION
# ---------------------------------------------------------------------------

def generate_analytics_data(
    repos: list[dict[str, Any]],
    skills: dict[str, Any],
    scores: dict[str, Any],
) -> dict[str, Any]:
    empty = {
        "activity_timeline":    {},
        "skill_distribution":   {},
        "intelligence_breakdown": {},
        "tech_evolution":       {},
    }

    if not repos:
        return empty

    # Activity timeline — repos created per year
    activity_timeline: dict[str, int] = {}
    for repo in repos:
        created = repo.get("created_at")
        if created:
            year = str(created.year)
            activity_timeline[year] = activity_timeline.get(year, 0) + 1

    # Skill distribution
    skill_distribution = skills.get("skills", {})

    # Intelligence breakdown
    intelligence_breakdown = scores.get("breakdown", {})

    # Tech evolution — languages appearing per year (using created_at)
    tech_evolution: dict[str, list[str]] = {}
    for repo in repos:
        lang    = repo.get("language")
        created = repo.get("created_at")
        if lang and created:
            year = str(created.year)
            if year not in tech_evolution:
                tech_evolution[year] = []
            if lang not in tech_evolution[year]:
                tech_evolution[year].append(lang)

    # Make cumulative — each year shows all languages seen so far
    sorted_years = sorted(tech_evolution.keys())
    seen: set[str] = set()
    cumulative: dict[str, list[str]] = {}
    for year in sorted_years:
        seen.update(tech_evolution[year])
        cumulative[year] = sorted(seen)

    return {
        "activity_timeline":      activity_timeline,
        "skill_distribution":     skill_distribution,
        "intelligence_breakdown": intelligence_breakdown,
        "tech_evolution":         cumulative,
    }


# ---------------------------------------------------------------------------
# MERGE + CONFIDENCE
# ---------------------------------------------------------------------------

def _compute_confidence(repos: list[dict[str, Any]]) -> str:
    total = len(repos)
    if total >= CONFIDENCE_HIGH_MIN:
        return "HIGH"
    elif total >= CONFIDENCE_MEDIUM_MIN:
        return "MEDIUM"
    return "LOW"


def merge_outputs(
    skills: dict[str, Any],
    behavior: dict[str, Any],
    profile: dict[str, Any],
    scores: dict[str, Any],
    analytics: dict[str, Any],
    repos: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "skill_evidence": {
            "skills":      skills.get("skills", {}),
            "skill_score": skills.get("skill_score", 0),
        },
        "engineering_behavior": {
            "consistency":    behavior.get("consistency", 0),
            "exploration":    behavior.get("exploration", 0),
            "depth":          behavior.get("depth", 0),
            "behavior_score": behavior.get("behavior_score", 0),
        },
        "developer_profile": {
            "archetype":         profile.get("archetype", "Insufficient Data"),
            "learning_velocity": profile.get("learning_velocity", "LOW"),
            "potential_score":   profile.get("potential_score", 0),
        },
        "github_engineering_intelligence": {
            "overall_score": scores.get("overall_score", 0),
            "breakdown":     scores.get("breakdown", {}),
            "strengths":     scores.get("strengths", []),
            "risks":         scores.get("risks", []),
            "confidence":    _compute_confidence(repos),
        },
        "analytics": analytics,
    }


# ---------------------------------------------------------------------------
# EMPTY RESULT — returned when no repos found
# ---------------------------------------------------------------------------

def _empty_result() -> dict[str, Any]:
    return {
        "skill_evidence": {"skills": {}, "skill_score": 0},
        "engineering_behavior": {
            "consistency": 0, "exploration": 0,
            "depth": 0, "behavior_score": 0,
        },
        "developer_profile": {
            "archetype": "Insufficient Data",
            "learning_velocity": "LOW",
            "potential_score": 0,
        },
        "github_engineering_intelligence": {
            "overall_score": 0,
            "breakdown": {},
            "strengths": [],
            "risks": ["Insufficient GitHub data"],
            "confidence": "LOW",
        },
        "analytics": {
            "activity_timeline": {},
            "skill_distribution": {},
            "intelligence_breakdown": {},
            "tech_evolution": {},
        },
    }


# ---------------------------------------------------------------------------
# ORCHESTRATOR
# ---------------------------------------------------------------------------

def analyze_github(username: str) -> dict[str, Any]:
    raw_repos = fetch_repositories(username)
    repos     = normalize_repo_data(raw_repos)

    if not repos:
        return _empty_result()

    skills    = analyze_skill_evidence(repos)
    behavior  = analyze_engineering_behavior(repos)
    profile   = classify_developer_profile(repos, behavior, skills)
    scores    = compute_engineering_scores(skills, behavior, profile, len(repos))
    analytics = generate_analytics_data(repos, skills, scores)

    return merge_outputs(skills, behavior, profile, scores, analytics, repos)

# ---------------------------------------------------------------------------
# MAIN RUNNER (TEST EXECUTION)
# ---------------------------------------------------------------------------

def main():

    print("\nGitHub Engineering Intelligence Analyzer\n")

    username = input("Enter GitHub username: ").strip()

    if not username:
        print("Invalid username")
        return

    print("\nAnalyzing profile...\n")

    result = analyze_github(username)

    intel = result["github_engineering_intelligence"]
    profile = result["developer_profile"]
    behavior = result["engineering_behavior"]
    skills = result["skill_evidence"]

    print("===== ENGINEERING INTELLIGENCE REPORT =====\n")

    print(f"Overall Score: {intel['overall_score']}/100")
    print(f"Confidence: {intel['confidence']}")
    print(f"Archetype: {profile['archetype']}")
    print(f"Learning Velocity: {profile['learning_velocity']}")

    print("\n--- Behavior Signals ---")

    print(f"Consistency: {behavior['consistency']}")
    print(f"Exploration: {behavior['exploration']}")
    print(f"Depth: {behavior['depth']}")

    print("\n--- Top Skills ---")

    top_skills = sorted(
        skills["skills"].items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    for skill, score in top_skills:
        print(f"{skill}: {score}")

    print("\n--- Strengths ---")

    for s in intel["strengths"]:
        print(f"+ {s}")

    print("\n--- Risks ---")

    for r in intel["risks"]:
        print(f"- {r}")

    print("\n==========================================\n")


if __name__ == "__main__":
    main()