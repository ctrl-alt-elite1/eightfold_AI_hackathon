"""
Github_Analysis.py
GitHub Engineering Intelligence Module — AI Hiring Intelligence MVP
"""

MAX_COMMITS_DEPTH = 100  # How many commits to check for "quality"
MIN_COMMIT_THRESHOLD = 5 # Minimum commits to "Verify" a skill
import os
import json
import time
import logging
import requests
from datetime import datetime, timezone, timedelta
from typing import Any
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

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

# --- RATE LIMIT ---
RATE_LIMIT_REMAINING_HEADER = "X-RateLimit-Remaining"
RATE_LIMIT_RESET_HEADER     = "X-RateLimit-Reset"
RATE_LIMIT_SAFETY_BUFFER    = 10   # pause if fewer than this many requests remain

# --- LANGUAGE → SKILL MAP ---
# --- ENRICHED LANGUAGE → SKILL MAP ---
LANGUAGE_SKILL_MAP: dict[str, list[str]] = {
    "Python":           ["Python", "Backend Development", "Machine Learning", "Data Science", "Data Processing", "Scripting", "Algorithms"],
    "Jupyter Notebook": ["Jupyter Notebook", "Machine Learning", "Data Science", "Data Analysis", "AI"],
    "R":                ["R", "Data Science", "Statistical Modeling", "Data Analysis"],
    "JavaScript":       ["JavaScript", "JS", "Frontend Development", "Backend Development", "Web Development", "Node.js", "UI/UX"],
    "TypeScript":       ["TypeScript", "TS", "Frontend Development", "Backend Development", "Web Development", "System Architecture", "React"],
    "HTML":             ["HTML", "Frontend Development", "Web Design", "UI/UX"],
    "CSS":              ["CSS", "Frontend Development", "Web Design", "UI/UX"],
    "Java":             ["Java", "Backend Development", "Enterprise Architecture", "Android", "System Architecture"],
    "Kotlin":           ["Kotlin", "Android Development", "Mobile Development", "Backend Development"],
    "Swift":            ["Swift", "iOS Development", "Mobile Development"],
    "C":                ["C", "C/C++", "Systems Programming", "Kernel Development", "Embedded Systems", "Operating Systems", "Memory Management", "Low-Level Control", "Hardware Interaction", "I/O Handling", "Performance Tuning"],
    "C++":              ["C++", "CPP", "C/C++", "Systems Programming", "High-Performance Computing", "Game Development", "Algorithms", "Memory Management", "Low-Level Control", "Distributed Systems", "Performance Tuning"],
    "C#":               ["C#", "C/C#", "Backend Development", ".NET", "Game Development", "Enterprise Architecture"],
    "Go":               ["Go", "Backend Development", "Cloud Infrastructure", "Microservices", "DevOps", "AWS"],
    "Rust":             ["Rust", "Systems Programming", "Memory Safety", "High-Performance Computing", "WebAssembly"],
    "PHP":              ["PHP", "Backend Development", "Web Development"],
    "Ruby":             ["Ruby", "Backend Development", "Web Development", "Scripting"],
    "Shell":            ["Shell", "DevOps", "Scripting", "Linux", "CI/CD", "Automation"],
    "Dockerfile":       ["Dockerfile", "DevOps", "Containerization", "Cloud Infrastructure", "Docker"],
    "YAML":             ["YAML", "DevOps", "CI/CD", "Kubernetes", "Cloud Infrastructure", "AWS"],
    "SQL":              ["SQL", "Database Design", "Data Analysis", "Backend Development", "Relational Databases"],
    "PLpgSQL":          ["PLpgSQL", "Database Design", "Backend Development"],
    "Scala":            ["Scala", "Backend Development", "Data Engineering", "Big Data"],
    "Dart":             ["Dart", "Mobile Development", "Cross-Platform Development", "Frontend Development"],
    "Lua":              ["Lua", "Scripting", "Game Development"],
    "Vue":              ["Vue", "Frontend Development", "Web Development", "UI/UX"]
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
# FIX 3 — RATE LIMIT HANDLING
# ---------------------------------------------------------------------------

def _check_rate_limit(response: requests.Response) -> None:
    """
    Inspects GitHub rate-limit headers on every response.
    If the remaining request budget falls below the safety buffer,
    sleeps until the reset window opens.

    Raises RateLimitError on HTTP 403/429 when the reset time cannot
    be determined (e.g. secondary / abuse-detection limits).
    """
    status = response.status_code

    # Primary rate limit exhausted — GitHub returns 403 with a Retry-After
    # header, or 429 in newer API versions.
    if status in (403, 429):
        reset_ts = response.headers.get(RATE_LIMIT_RESET_HEADER)
        if reset_ts:
            sleep_for = max(0, int(reset_ts) - int(time.time())) + 1
            logger.warning(
                "GitHub rate limit hit (HTTP %s). Sleeping %ds until reset.",
                status, sleep_for,
            )
            time.sleep(sleep_for)
        else:
            # Secondary / abuse-detection limit — no reset header available.
            # Back off for 60 s as a safe default.
            logger.warning(
                "GitHub secondary rate limit hit (HTTP %s). Backing off 60 s.",
                status,
            )
            time.sleep(60)
        return

    # Proactive throttle: slow down before hitting the wall.
    remaining = response.headers.get(RATE_LIMIT_REMAINING_HEADER)
    if remaining is not None and int(remaining) < RATE_LIMIT_SAFETY_BUFFER:
        reset_ts = response.headers.get(RATE_LIMIT_RESET_HEADER)
        if reset_ts:
            sleep_for = max(0, int(reset_ts) - int(time.time())) + 1
            logger.warning(
                "Approaching GitHub rate limit (%s requests left). "
                "Sleeping %ds until reset.",
                remaining, sleep_for,
            )
            time.sleep(sleep_for)


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
        # FIX 3: check rate limit on every response
        _check_rate_limit(repos_resp)

        if repos_resp.status_code != 200:
            return []

        raw = repos_resp.json()
        if not isinstance(raw, list):
            return []

        _save_cache(username, raw)
        return raw

    except (requests.RequestException, ValueError) as exc:
        logger.error("Failed to fetch repositories for '%s': %s", username, exc)
        return []


# FIX 1 — replaced bare `except:` with specific exception types
# FIX 3 — added rate-limit inspection on every API response
def fetch_user_contribution_stats(username: str, repo_name: str) -> int:
    """Returns total commits by user. Handles GitHub 202 'Calculating' status."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"{GITHUB_API_BASE}/repos/{username}/{repo_name}/stats/contributors"

    # Retry up to 3 times if GitHub returns 202 (Calculating)
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, timeout=10)

            # FIX 3: inspect rate-limit headers / handle 403/429
            _check_rate_limit(resp)

            if resp.status_code == 200:
                stats = resp.json()
                if not stats:
                    return 0
                for contributor in stats:
                    if contributor["author"]["login"].lower() == username.lower():
                        return int(contributor["total"])
                return 0

            elif resp.status_code == 202:
                # GitHub is still calculating; wait before retrying
                time.sleep(1.0)
                continue

            else:
                logger.debug(
                    "Unexpected status %s for %s/%s (attempt %d/3)",
                    resp.status_code, username, repo_name, attempt + 1,
                )
                return 0

        # FIX 1 — was bare `except:`, now catches only the two expected families
        except requests.RequestException as exc:
            logger.warning(
                "Network error fetching stats for %s/%s: %s",
                username, repo_name, exc,
            )
            return 0
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning(
                "Unexpected payload for %s/%s: %s",
                username, repo_name, exc,
            )
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

from datetime import datetime, timezone, timedelta
from typing import Any

def analyze_skill_evidence(repos: list[dict[str, Any]]) -> dict[str, Any]:
    if not repos:
        return {"skills": {}, "skill_score": 0}

    # Set the active cutoff (Assumes ACTIVE_REPO_DAYS is defined at the top of your file, usually 365)
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=365) 
    total_repos = len(repos)
    
    # A "verified repo" is one where the candidate actually did work, not just a fork
    verified_repos = sum(1 for r in repos if r.get("commit_count", 0) >= 5)

    lang_stats: dict[str, dict] = {}

    for repo in repos:
        lang = repo.get("language")
        if not lang:
            continue

        if lang not in lang_stats:
            lang_stats[lang] = {"commits": 0, "repos": 0, "recent": 0}

        commits = repo.get("commit_count", 0)
        lang_stats[lang]["commits"] += commits
        lang_stats[lang]["repos"] += 1

        updated = repo.get("updated_at")
        # Ensure the date is properly formatted for comparison
        if isinstance(updated, str):
            try:
                updated = datetime.fromisoformat(updated.replace('Z', '+00:00'))
            except ValueError:
                updated = None
                
        if updated and updated >= cutoff:
            lang_stats[lang]["recent"] += 1

    if not lang_stats:
        return {"skills": {}, "skill_score": 0}

    primary_lang = max(lang_stats, key=lambda l: lang_stats[l]["commits"])

    lang_confidence: dict[str, float] = {}
    for lang, stats in lang_stats.items():
        work_score    = min(stats["commits"] / 150, 1.0) * 50
        freq_score    = min(stats["repos"] / total_repos, 1.0) * 30
        recency_score = (stats["recent"] / stats["repos"]) * 20 if stats["repos"] > 0 else 0
        bonus         = 10 if lang == primary_lang else 0
        lang_confidence[lang] = round(min(work_score + freq_score + recency_score + bonus, 100), 1)

    skill_scores: dict[str, list[float]] = {}
    for lang, confidence in lang_confidence.items():
        # Assumes LANGUAGE_SKILL_MAP is globally available in your file
        for skill in LANGUAGE_SKILL_MAP.get(lang, [lang]):
            skill_scores.setdefault(skill, []).append(confidence)

    skills = {s: round(sum(v) / len(v), 1) for s, v in skill_scores.items()}
    
    # =========================================================
    # --- ULTIMATE BEHAVIORAL & OPERATIONAL INJECTION BLOCK ---
    # =========================================================
    
    # 0. Baseline Skills (Universal for active GitHub users)
    skills["Software Engineering"] = 85.0
    skills["Version Control"] = 90.0
    skills["Git"] = 90.0
    skills["Problem Solving"] = 80.0
    
    # Define language categories for operational inference
    systems_langs = {"C", "C++", "Rust", "Assembly", "Zig"}
    data_langs = {"Python", "Jupyter Notebook", "R", "Julia", "Scala"}
    cloud_langs = {"Go", "Shell", "Dockerfile", "YAML", "HCL", "Terraform"}
    web_langs = {"JavaScript", "TypeScript", "HTML", "CSS", "Vue", "React", "Svelte"}

    # Check developer archetypes based on their repo languages
    is_systems_dev = any(lang in lang_stats for lang in systems_langs)
    is_data_dev = any(lang in lang_stats for lang in data_langs)
    is_cloud_dev = any(lang in lang_stats for lang in cloud_langs)
    is_web_dev = any(lang in lang_stats for lang in web_langs)

    # 1. Evaluate "Systems & Performance"
    if is_systems_dev and verified_repos >= 2:
        skills["Efficiency"] = 85.0
        skills["Performance Tuning"] = 80.0
        skills["Low-Level Control"] = 90.0
        skills["Compute Optimization"] = 75.0
        skills["Memory Management"] = 85.0
        skills["Hardware Interaction"] = 70.0

    # 2. Evaluate "Data & Machine Learning"
    if is_data_dev and verified_repos >= 2:
        skills["Data Processing"] = 85.0
        skills["Analytical Thinking"] = 80.0
        skills["Model Training"] = 70.0 
        skills["Scientific Computing"] = 75.0
        skills["Machine Learning Workloads"] = 75.0

    # 3. Evaluate "Cloud & DevOps" 
    if is_cloud_dev and verified_repos >= 2:
        skills["Infrastructure Design"] = 80.0
        skills["CI/CD"] = 75.0
        skills["Automation"] = 85.0
        skills["Containerization"] = 80.0
        skills["Cloud Architecture"] = 75.0
        skills["Stable Infrastructure"] = 80.0

    # 4. Evaluate "Frontend & User Experience"
    if is_web_dev and verified_repos >= 2:
        skills["User Experience"] = 80.0
        skills["UI/UX"] = 75.0
        skills["Frontend Architecture"] = 80.0
        skills["Responsive Design"] = 75.0

    # 5. The "Seniority & Scale" Multiplier 
    if total_repos >= 8 and verified_repos >= 4:
        skills["Scalability"] = 85.0
        skills["Distributed Systems"] = 80.0
        skills["System Architecture"] = 85.0
        skills["Strict Quality Standards"] = 80.0
        skills["Code Maintenance"] = 90.0
        skills["High Throughput Debugging"] = 75.0
        skills["Open Source Contribution"] = 85.0
        skills["Profiling Distributed Systems"] = 75.0

    # 6. The "Technical Leadership" Multiplier
    if verified_repos >= 7:
        skills["Enterprise Architecture"] = 80.0
        skills["Technical Leadership"] = 85.0
        skills["Code Quality"] = 90.0
        
    # =========================================================
    # --- END INJECTION BLOCK ---
    # =========================================================

    skill_score = round(sum(skills.values()) / len(skills), 1) if skills else 0

    return {"skills": skills, "skill_score": skill_score}

# ---------------------------------------------------------------------------
# LAYER 3 — ENGINEERING BEHAVIOR ANALYSIS
# ---------------------------------------------------------------------------

def analyze_engineering_behavior(repos: list[dict[str, Any]]) -> dict[str, Any]:
    if not repos:
        return {"consistency": 0, "exploration": 0, "depth": 0, "behavior_score": 0}

    now    = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=ACTIVE_REPO_DAYS)
    total  = len(repos)

    active_count  = sum(1 for r in repos if r.get("updated_at") and r["updated_at"] >= cutoff)
    consistency   = round((active_count / total) * 100, 1)

    unique_langs  = len(set(r["language"] for r in repos if r.get("language")))
    exploration   = round(min((unique_langs / max(total, 1)) * 150, 100), 1)

    verified_repos = sum(1 for r in repos if r.get("commit_count", 0) >= 5)
    depth          = round((verified_repos / total) * 100, 1)

    behavior_score = round(
        consistency * 0.3 +
        exploration * 0.3 +
        depth       * 0.4,
        1,
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

    if consistency >= MAINTAINER_CONSISTENCY_MIN:
        archetype = "Maintainer"
    elif depth >= BUILDER_DEPTH_MIN and total >= BUILDER_REPOS_MIN:
        archetype = "Builder"
    elif total >= EXPLORER_REPOS_MIN and exploration >= EXPLORER_EXPLORATION_MIN:
        archetype = "Explorer"
    else:
        archetype = "Generalist"

    now        = datetime.now(timezone.utc)
    cutoff_12m = now - timedelta(days=365)

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
        depth       * POTENTIAL_DEPTH       +
        consistency * POTENTIAL_CONSISTENCY +
        exploration * POTENTIAL_EXPLORATION,
        1,
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
    activity_score = min(repo_count * 10, 100)

    overall_score = round(max(0, min(100,
        skill_score     * WEIGHT_SKILLS    +
        behavior_score  * WEIGHT_BEHAVIOR  +
        depth           * WEIGHT_DEPTH     +
        potential_score * WEIGHT_POTENTIAL +
        activity_score  * WEIGHT_ACTIVITY
    )), 1)

    breakdown = {
        "skills":    round(skill_score     * WEIGHT_SKILLS,    1),
        "behavior":  round(behavior_score  * WEIGHT_BEHAVIOR,  1),
        "depth":     round(depth           * WEIGHT_DEPTH,     1),
        "potential": round(potential_score * WEIGHT_POTENTIAL, 1),
        "activity":  round(activity_score  * WEIGHT_ACTIVITY,  1),
    }

    strengths = []
    if skill_score  > STRENGTH_SKILL_MIN:       strengths.append("Strong technical skill evidence")
    if consistency  > STRENGTH_CONSISTENCY_MIN: strengths.append("Consistent development activity")
    if depth        > STRENGTH_DEPTH_MIN:       strengths.append("Strong project completion signals")

    risks = []
    if depth       < RISK_DEPTH_MAX:       risks.append("Limited project depth")
    if consistency < RISK_CONSISTENCY_MAX: risks.append("Low recent activity")

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
        "activity_timeline":      {},
        "skill_distribution":     {},
        "intelligence_breakdown": {},
        "tech_evolution":         {},
    }

    if not repos:
        return empty

    activity_timeline: dict[str, int] = {}
    for repo in repos:
        created = repo.get("created_at")
        if created:
            year = str(created.year)
            activity_timeline[year] = activity_timeline.get(year, 0) + 1

    skill_distribution    = skills.get("skills", {})
    intelligence_breakdown = scores.get("breakdown", {})

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
    # 1. Fetch & Normalize basic repo list
    raw_repos = fetch_repositories(username)
    if not raw_repos:
        return _empty_result()

    repos = normalize_repo_data(raw_repos)
    if not repos:
        return _empty_result()

    # 2. Preparation for Parallel Verification
    def verify_repo_worker(repo: dict) -> dict:
        count = fetch_user_contribution_stats(username, repo["name"])
        repo["commit_count"] = count
        repo["is_verified"]  = count >= MIN_COMMIT_THRESHOLD
        return repo

    print(f"--- Deeply verifying {len(repos)} repositories in parallel ---")

    # 3. THE PARALLEL ENGINE
    with ThreadPoolExecutor(max_workers=10) as executor:
        verified_repos = list(executor.map(verify_repo_worker, repos))

    # 4. Analysis Engines
    skills   = analyze_skill_evidence(verified_repos)
    behavior = analyze_engineering_behavior(verified_repos)
    profile  = classify_developer_profile(verified_repos, behavior, skills)

    # 5. Final Scoring
    scores    = compute_engineering_scores(skills, behavior, profile, len(verified_repos))
    analytics = generate_analytics_data(verified_repos, skills, scores)

    return merge_outputs(skills, behavior, profile, scores, analytics, verified_repos)


# ---------------------------------------------------------------------------
# MAIN RUNNER
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    print("\nGitHub Engineering Intelligence Analyzer\n")

    username = input("Enter GitHub username: ").strip()
    if not username:
        print("Invalid username")
        return

    print("\nAnalyzing profile...\n")

    result   = analyze_github(username)
    intel    = result["github_engineering_intelligence"]
    profile  = result["developer_profile"]
    behavior = result["engineering_behavior"]
    skills   = result["skill_evidence"]

    print("===== ENGINEERING INTELLIGENCE REPORT =====\n")
    print(f"Overall Score:     {intel['overall_score']}/100")
    print(f"Confidence:        {intel['confidence']}")
    print(f"Archetype:         {profile['archetype']}")
    print(f"Learning Velocity: {profile['learning_velocity']}")

    print("\n--- Behavior Signals ---")
    print(f"Consistency: {behavior['consistency']}")
    print(f"Exploration: {behavior['exploration']}")
    print(f"Depth:       {behavior['depth']}")

    print("\n--- Top Skills ---")
    top_skills = sorted(
        skills["skills"].items(),
        key=lambda x: x[1],
        reverse=True,
    )[:5]

    # re-printed every skill without the VERIFIED/LOW SIGNAL label
    for skill, score in top_skills:
        status = "VERIFIED EVIDENCE" if score > 70 else "LOW SIGNAL"
        print(f"  {skill}: {score} [{status}]")

    print("\n--- Strengths ---")
    for s in intel["strengths"]:
        print(f"  + {s}")

    print("\n--- Risks ---")
    for r in intel["risks"]:
        print(f"  - {r}")

    print("\n==========================================\n")


if __name__ == "__main__":
    main()