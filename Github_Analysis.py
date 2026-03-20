"""
Github_Analysis.py
GitHub Engineering Intelligence Module — AI Hiring Intelligence MVP
"""

MAX_COMMITS_DEPTH = 100  # How many commits to check for "quality"
MIN_COMMIT_THRESHOLD = 5 # Minimum commits to "Verify" a skill
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


def fetch_user_contribution_stats(username: str, repo_name: str) -> int:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"{GITHUB_API_BASE}/repos/{username}/{repo_name}/stats/contributors"
    
    # Retry logic: Try up to 3 times if we get a 202
    for _ in range(3):
        resp = requests.get(url, headers=headers, timeout=10)
        
        if resp.status_code == 200:
            stats = resp.json()
            for contributor in stats:
                if contributor['author']['login'].lower() == username.lower():
                    return int(contributor['total'])
            return 0
            
        elif resp.status_code == 202:
            # GitHub is calculating. Wait a bit and try again.
            time.sleep(1.0) 
            continue
            
        else:
            return 0
    return 0

def normalize_repo_data(raw_repos: list[dict]) -> list[dict[str, Any]]:
    """Clean and format raw GitHub API data."""
    if not raw_repos or not isinstance(raw_repos, list):
        return []

    normalized = []
    for repo in raw_repos:
        if not isinstance(repo, dict):
            continue
        # Skip forks (as per the PDF 'High-Trust' requirement)
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
            # We don't fetch commit_count here anymore!
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
    total_repos = len(repos)

    # Accumulators
    lang_stats: dict[str, dict] = {} # { "Python": {"commits": 0, "repos": 0, "recent": 0} }

    for repo in repos:
        lang = repo.get("language")
        if not lang: continue
        
        if lang not in lang_stats:
            lang_stats[lang] = {"commits": 0, "repos": 0, "recent": 0}
        
        commits = repo.get("commit_count", 0)
        lang_stats[lang]["commits"] += commits
        lang_stats[lang]["repos"] += 1
        
        updated = repo.get("updated_at")
        if updated and updated >= cutoff:
            lang_stats[lang]["recent"] += 1

    if not lang_stats:
        return {"skills": {}, "skill_score": 0}

    # Identify primary language by commit volume, not just repo count
    primary_lang = max(lang_stats, key=lambda l: lang_stats[l]["commits"])

    lang_confidence: dict[str, float] = {}
    for lang, stats in lang_stats.items():
        # 1. Workload Score (Volume of Code): 50 pts
        # Benchmarked at 50 commits for "Mastery" in a hackathon context
        work_score = min(stats["commits"] / 50, 1.0) * 50
        
        # 2. Consistency Score (Spread across projects): 30 pts
        freq_score = min(stats["repos"] / total_repos, 1.0) * 30
        
        # 3. Recency Score (Active usage): 20 pts
        recency_score = (stats["recent"] / stats["repos"]) * 20
        
        # Bonus for the language they use the most
        bonus = 10 if lang == primary_lang else 0
        
        lang_confidence[lang] = round(min(work_score + freq_score + recency_score + bonus, 100), 1)

    # Map to professional skills (Existing logic)
    skill_scores: dict[str, list[float]] = {}
    for lang, confidence in lang_confidence.items():
        for skill in LANGUAGE_SKILL_MAP.get(lang, [lang]):
            skill_scores.setdefault(skill, []).append(confidence)

    skills = {s: round(sum(v)/len(v), 1) for s, v in skill_scores.items()}
    skill_score = round(sum(skills.values()) / len(skills), 1) if skills else 0

    return {"skills": skills, "skill_score": skill_score}


# ---------------------------------------------------------------------------
# LAYER 3 — ENGINEERING BEHAVIOR ANALYSIS
# ---------------------------------------------------------------------------

def analyze_engineering_behavior(repos: list[dict[str, Any]]) -> dict[str, Any]:
    if not repos:
        return {"consistency": 0, "exploration": 0, "depth": 0, "behavior_score": 0}

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=ACTIVE_REPO_DAYS)
    total = len(repos)

    # 1. Consistency: How many repos were updated recently?
    active_count = sum(1 for r in repos if r.get("updated_at") and r["updated_at"] >= cutoff)
    consistency = round((active_count / total) * 100, 1)

    # 2. Exploration: Variety of languages used
    unique_langs = len(set(r["language"] for r in repos if r.get("language")))
    exploration = round(min(unique_langs * 15, 100), 1)

    # 3. NEW Depth: Percentage of repos that are "Verified" (5+ commits)
    # This proves the candidate builds things rather than just forking/starring.
    verified_repos = sum(1 for r in repos if r.get("commit_count", 0) >= 5)
    depth = round((verified_repos / total) * 100, 1)

    # Calculate overall behavior score with new weights
    behavior_score = round(
        consistency * 0.3 + 
        exploration * 0.3 + 
        depth * 0.4, # Weight depth slightly higher for "High Trust"
        1
    )

    return {
        "consistency": consistency,
        "exploration": exploration,
        "depth": depth,
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
    # 1. Fetch raw list
    raw_repos = fetch_repositories(username)
    if not raw_repos:
        return _empty_result()

    # 2. Normalize (No username needed here anymore)
    repos = normalize_repo_data(raw_repos)
    if not repos:
        return _empty_result()

    # 3. Sort by recency (Standard Hackathon Strategy)
    repos = sorted(
        repos, 
        key=lambda x: x.get('updated_at') or datetime.min.replace(tzinfo=timezone.utc), 
        reverse=True
    )
    
    # 4. Deep Verification (Where the 'username' is used)
    top_repos = repos[:5]
    other_repos = repos[5:]

    for repo in top_repos:
        # We have the username here, so no NameError!
        repo['commit_count'] = fetch_user_contribution_stats(username, repo['name'])
        repo['is_verified'] = repo['commit_count'] >= 5
        
    for repo in other_repos:
        repo['commit_count'] = 1  # Standard weight for non-verified repos
        repo['is_verified'] = False

    verified_repos = top_repos + other_repos

    # 5. Pass the verified data to your analysis engines
    skills    = analyze_skill_evidence(verified_repos)
    behavior  = analyze_engineering_behavior(verified_repos)
    profile   = classify_developer_profile(verified_repos, behavior, skills)
    scores    = compute_engineering_scores(skills, behavior, profile, len(verified_repos))
    analytics = generate_analytics_data(verified_repos, skills, scores)

    return merge_outputs(skills, behavior, profile, scores, analytics, verified_repos)
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
    
    # Add a "Verified" badge in the UI/Terminal
    for skill, score in top_skills:
        status = "VERIFIED EVIDENCE" if score > 70 else "LOW SIGNAL"
        print(f"{skill}: {score} [{status}]")

    print("\n==========================================\n")
    


if __name__ == "__main__":
    main()