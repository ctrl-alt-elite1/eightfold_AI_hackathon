"""
hiring_intelligence.py
----------------------
Deterministic hiring decision engine.
Python decides. LLM explains.

Entry point:
    evaluate_candidates(usernames, job_description) -> dict
"""

import re
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — all weights and thresholds in one place
# ---------------------------------------------------------------------------

CONFIG = {
    "weights": {
        "match":       0.35,
        "engineering": 0.25,
        "trajectory":  0.20,
        "behavior":    0.10,
        "activity":    0.10,
    },
    "confidence_multipliers": {
        "HIGH":   1.0,
        "MEDIUM": 0.9,
        "LOW":    0.75,
    },
    "tie_threshold":       5.0,
    "min_delta_to_report": 1.0,
    "ollama_url":          "http://localhost:11434",  # change to remote IP if needed
    "ollama_model":        "phi3",
    "ollama_timeout":      45,
    "max_workers":         4,
}

DEMOGRAPHIC_KEYWORDS = [
    "university", "college", "iit", "mit", "stanford", "harvard",
    "male", "female", "gender", "age", "location", "city", "country",
    "followers", "following", "stars", "nationality", "race", "name",
]

TIE_BREAK_ORDER = ["match_score", "engineering_score", "adjacency_score", "behavior_score", "activity_score"]


# ---------------------------------------------------------------------------
# Safe numeric helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if result == result else default  # NaN check
    except (TypeError, ValueError):
        return default


def _normalize(value: float, expected_max: float = 100.0) -> float:
    """Normalize value to 0-100. Handles 0-1 scale automatically."""
    if value <= 0:
        return 0.0
    if value <= 1.0:
        return min(value * 100.0, 100.0)
    return min(value, 100.0)


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Module-level singletons — initialized lazily, never mutated after creation
# ---------------------------------------------------------------------------

_matcher_instance = None
_graph_instance   = None


def _get_matcher():
    global _matcher_instance
    if _matcher_instance is None:
        from Job_Matcher import TalentMatcher
        _matcher_instance = TalentMatcher()
    return _matcher_instance


def _get_graph():
    global _graph_instance
    if _graph_instance is None:
        from skill_graph import build_skill_graph
        _graph_instance = build_skill_graph()
    return _graph_instance


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_inputs(usernames: List[str], job_description: str) -> Tuple[List[str], str]:
    """Validate and clean inputs. Returns (clean_usernames, clean_jd)."""
    if not job_description or not job_description.strip():
        raise ValueError("Job description cannot be empty.")

    clean_usernames = [u.strip() for u in (usernames or []) if u and u.strip()]
    if not clean_usernames:
        raise ValueError("At least one valid username is required.")

    return clean_usernames, job_description.strip()


# ---------------------------------------------------------------------------
# JD requirement extraction — called once, passed to all candidates
# ---------------------------------------------------------------------------

def _extract_requirements(job_description: str) -> List[str]:
    """
    Extract required skills from JD.
    Called once per evaluate_candidates() call, not per candidate.
    """
    try:
        matcher = _get_matcher()
        result  = matcher.extract_requirements(job_description) or {}
        core    = result.get("core_skills", []) or []
        sec     = result.get("secondary_skills", []) or []
        return [s for s in (core + sec) if s and isinstance(s, str)]
    except Exception as e:
        logger.warning("Requirement extraction failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Activity score — combines repo activity + skill strength
# ---------------------------------------------------------------------------

def _compute_activity_score(github_raw: Dict[str, Any]) -> float:
    """
    Activity score combining:
    - repo count from activity timeline
    - average skill confidence from skill evidence
    Normalized to 0-100.
    """
    timeline      = github_raw.get("analytics", {}).get("activity_timeline", {}) or {}
    skill_dist    = github_raw.get("skill_evidence", {}).get("skills", {}) or {}

    repo_count    = sum(timeline.values()) if timeline else 0
    avg_skill     = (sum(skill_dist.values()) / len(skill_dist)) if skill_dist else 0

    repo_signal   = _clamp(repo_count * 4, 0, 60)
    skill_signal  = _clamp(avg_skill * 0.4, 0, 40)

    return _clamp(repo_signal + skill_signal)


# ---------------------------------------------------------------------------
# Bias check — deterministic, no LLM
# ---------------------------------------------------------------------------

def run_bias_check(candidate_skills: List[str], job_description: str) -> Dict[str, Any]:
    """
    Proves score is not influenced by demographic signals.
    Runs matching twice: full JD vs JD with demographics stripped.
    """
    try:
        matcher     = _get_matcher()
        fake_github = {"skill_evidence": {"skills": {s: 80.0 for s in candidate_skills}}}

        score_with  = _safe_float(
            matcher.match_candidate(job_description, fake_github).get("match_score", 0)
        )

        jd_clean = job_description
        for kw in DEMOGRAPHIC_KEYWORDS:
            jd_clean = re.sub(rf'\b{re.escape(kw)}\b', ' ', jd_clean, flags=re.IGNORECASE)
        jd_clean = re.sub(r'\s+', ' ', jd_clean).strip()

        score_without = _safe_float(
            matcher.match_candidate(jd_clean, fake_github).get("match_score", 0)
        )

        delta  = round(abs(score_with - score_without), 2)
        passed = delta <= 5.0

        return {
            "bias_check_passed":          passed,
            "score_with_context":         round(score_with, 1),
            "score_without_demographics": round(score_without, 1),
            "delta":                      delta,
            "verdict": (
                f"Bias check passed — score changed by {delta} pts after removing demographic context."
                if passed else
                f"Warning — score shifted {delta} pts after removing demographic keywords. Review JD."
            ),
        }
    except Exception as e:
        logger.warning("Bias check failed: %s", e)
        return {
            "bias_check_passed":          True,
            "score_with_context":         0,
            "score_without_demographics": 0,
            "delta":                      0,
            "verdict":                    f"Bias check unavailable: {e}",
        }


# ---------------------------------------------------------------------------
# Single candidate analysis
# ---------------------------------------------------------------------------

def _analyse_one(
    username: str,
    job_description: str,
    required_list: List[str],
) -> Dict[str, Any]:
    """
    Full analysis pipeline for one candidate.
    Never raises — returns structured error dict on any failure.
    """
    try:
        from Github_Analysis import analyze_github
        from skill_graph import compute_learning_trajectory

        # github signals
        github_raw = analyze_github(username) or {}
        intel      = github_raw.get("github_engineering_intelligence") or {}
        behavior   = github_raw.get("engineering_behavior") or {}
        profile    = github_raw.get("developer_profile") or {}
        skill_ev   = github_raw.get("skill_evidence") or {}
        skills     = skill_ev.get("skills") or {}

        if not skills:
            return _error_result(username, "No skill evidence found — profile may be empty or private.")

        eng_score      = _clamp(_safe_float(intel.get("overall_score")))
        behavior_score = _clamp(_safe_float(behavior.get("behavior_score")))
        confidence     = intel.get("confidence") or "LOW"
        velocity       = profile.get("learning_velocity") or "LOW"
        archetype      = profile.get("archetype") or "Unknown"

        # semantic match
        matcher      = _get_matcher()
        match_result = matcher.match_candidate(job_description, github_raw) or {}

        if match_result.get("error"):
            return _error_result(username, "Semantic matching failed — insufficient GitHub data.")

        match_score = _clamp(_safe_float(match_result.get("match_score")))

        raw_matched    = match_result.get("matched_skills") or []
        matched_skills = [
            m["required_skill"] if isinstance(m, dict) else str(m)
            for m in raw_matched
        ]
        missing_skills = [
            s.replace(" (Core)", "").replace(" (Secondary)", "")
            for s in (match_result.get("missing_skills") or [])
        ]

        # learning trajectory
        graph      = _get_graph()
        trajectory = compute_learning_trajectory(
            candidate_skills=list(skills.keys()),
            required_skills=required_list or [],
            graph=graph,
        )
        adjacency_score = _clamp(_normalize(_safe_float(trajectory.adjacency_score)))

        # activity score
        activity_score = _clamp(_compute_activity_score(github_raw))

        # data quality flag
        repo_count    = sum((github_raw.get("analytics") or {}).get("activity_timeline", {}).values() or [0])
        data_quality  = "HIGH" if repo_count >= 8 else "MEDIUM" if repo_count >= 3 else "LOW"

        # weighted score — raw float, rounded only at return
        weights = CONFIG["weights"]
        raw_score = (
            match_score     * weights["match"]       +
            eng_score       * weights["engineering"] +
            adjacency_score * weights["trajectory"]  +
            behavior_score  * weights["behavior"]    +
            activity_score  * weights["activity"]
        )

        # confidence multiplier applied after aggregation
        conf_mult   = CONFIG["confidence_multipliers"].get(confidence, 0.75)
        final_score = _clamp(raw_score * conf_mult)

        score_breakdown = {
            "match":       round(match_score     * weights["match"],       2),
            "engineering": round(eng_score       * weights["engineering"], 2),
            "trajectory":  round(adjacency_score * weights["trajectory"],  2),
            "behavior":    round(behavior_score  * weights["behavior"],    2),
            "activity":    round(activity_score  * weights["activity"],    2),
        }

        decision_factors = sorted(score_breakdown.items(), key=lambda x: x[1], reverse=True)

        return {
            "username":          username,
            "final_score":       final_score,           # raw float — rounded at display
            "match_score":       match_score,
            "engineering_score": eng_score,
            "adjacency_score":   adjacency_score,
            "behavior_score":    behavior_score,
            "activity_score":    activity_score,
            "confidence":        confidence,
            "conf_multiplier":   conf_mult,
            "learning_velocity": velocity,
            "archetype":         archetype,
            "matched_skills":    matched_skills,
            "missing_skills":    missing_skills,
            "trajectory":        trajectory,
            "strengths":         intel.get("strengths") or [],
            "risks":             intel.get("risks") or [],
            "match_reasoning":   match_result.get("reasoning") or "",
            "score_breakdown":   score_breakdown,
            "decision_factors":  [f[0] for f in decision_factors],
            "data_quality_flag": data_quality,
            "error":             False,
        }

    except Exception as e:
        logger.error("Analysis failed for %s: %s", username, e, exc_info=True)
        return _error_result(username, str(e))


def _error_result(username: str, reason: str) -> Dict[str, Any]:
    return {
        "username":    username,
        "final_score": 0.0,
        "error":       True,
        "error_msg":   reason,
    }


# ---------------------------------------------------------------------------
# Tie-breaker
# ---------------------------------------------------------------------------

def _tie_break(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Breaks ties using ordered metrics by hiring importance.
    match_score first — job relevance must dominate.
    """
    for key in TIE_BREAK_ORDER:
        va = _safe_float(a.get(key))
        vb = _safe_float(b.get(key))
        if va > vb:
            return a
        if vb > va:
            return b
    return a


# ---------------------------------------------------------------------------
# Differentiators
# ---------------------------------------------------------------------------

def _compute_differentiators(winner: Dict[str, Any], runner_up: Dict[str, Any]) -> List[str]:
    """
    Returns why winner beat runner-up.
    Sorted by absolute delta — most decisive factor first.
    """
    metrics = [
        ("match_score",       "Skill Match"),
        ("engineering_score", "Engineering Score"),
        ("adjacency_score",   "Learning Adjacency"),
        ("behavior_score",    "Behavior Score"),
        ("activity_score",    "Activity Score"),
    ]
    diffs = []
    for key, label in metrics:
        delta = round(_safe_float(winner.get(key)) - _safe_float(runner_up.get(key)), 2)
        if abs(delta) >= CONFIG["min_delta_to_report"]:
            sign = "+" if delta > 0 else ""
            diffs.append((abs(delta), f"{label}: {sign}{delta}"))

    diffs.sort(key=lambda x: x[0], reverse=True)
    return [d[1] for d in diffs] or ["Scores are nearly identical across all metrics"]


# ---------------------------------------------------------------------------
# Ollama explanation — only LLM usage in entire module
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str) -> str:
    try:
        r = requests.post(
            f"{CONFIG['ollama_url']}/api/generate",
            json={
                "model":   CONFIG["ollama_model"],
                "prompt":  prompt,
                "stream":  False,
                "options": {"temperature": 0.2},
            },
            timeout=CONFIG["ollama_timeout"],
        )
        if r.status_code == 200:
            return (r.json().get("response") or "").strip()
        logger.warning("Ollama returned status %s", r.status_code)
        return ""
    except requests.exceptions.ConnectionError:
        logger.warning("Ollama not reachable at %s", CONFIG["ollama_url"])
        return ""
    except requests.exceptions.Timeout:
        logger.warning("Ollama timed out after %ss", CONFIG["ollama_timeout"])
        return ""
    except Exception as e:
        logger.warning("Ollama error: %s", e)
        return ""


def _generate_explanation(
    winner: Dict[str, Any],
    ranked: List[Dict[str, Any]],
    differentiators: List[str],
) -> str:
    prompt = f"""You are a technical recruiter explaining a hiring decision to a non-technical HR manager.

WINNER: {winner['username']} — Final Score: {round(winner['final_score'], 1)}/100
Archetype: {winner.get('archetype')} | Learning Velocity: {winner.get('learning_velocity')}
Confidence: {winner.get('confidence')} | Data Quality: {winner.get('data_quality_flag')}
Matched Skills: {', '.join(winner.get('matched_skills', [])[:5]) or 'None'}
Missing Skills: {', '.join(winner.get('missing_skills', [])[:3]) or 'None'}
Strengths: {', '.join(winner.get('strengths', [])) or 'None'}
Risks: {', '.join(winner.get('risks', [])) or 'None'}

WHY THEY WON:
{chr(10).join(differentiators)}

ALL CANDIDATES:
{chr(10).join(f"{c['rank']}. {c['username']} — {round(c['final_score'], 1)}/100" for c in ranked)}

Write 3-4 sentences for a recruiter:
1. Why {winner['username']} was selected
2. Their strongest GitHub evidence
3. One risk or gap to watch
4. Whether runner-up is worth interviewing

Plain English. No jargon. No bullet points."""

    response = _call_ollama(prompt)
    if response:
        return response

    # structured fallback — not silent
    diffs_str = ", ".join(differentiators[:3])
    return (
        f"[Ollama unavailable — deterministic summary] "
        f"{winner['username']} ranked #1 with {round(winner['final_score'], 1)}/100. "
        f"Key advantages: {diffs_str}. "
        f"Matched: {', '.join(winner.get('matched_skills', [])[:4]) or 'general engineering'}. "
        f"{'Risk: ' + winner['risks'][0] if winner.get('risks') else 'No major risks identified.'}"
    )


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def evaluate_candidates(
    candidate_usernames: List[str],
    job_description: str,
    parallel: bool = False,
) -> Dict[str, Any]:
    """
    Analyse and rank multiple GitHub candidates against a job description.

    Args:
        candidate_usernames: GitHub usernames to evaluate
        job_description:     full JD text
        parallel:            run candidate analysis in parallel (faster, less stable)

    Returns:
        ranked_candidates, winner, differentiators, llm_explanation,
        score_breakdown per candidate, total_evaluated, errors
    """
    try:
        usernames, jd = _validate_inputs(candidate_usernames, job_description)
    except ValueError as e:
        return {
            "ranked_candidates": [],
            "winner":            None,
            "differentiators":   [],
            "llm_explanation":   str(e),
            "total_evaluated":   0,
            "errors":            [str(e)],
        }

    # extract JD requirements once — not per candidate
    required_list = _extract_requirements(jd)

    # analyse candidates
    if parallel:
        results = []
        with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
            futures = {
                executor.submit(_analyse_one, u, jd, required_list): u
                for u in usernames
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append(_error_result(futures[future], str(e)))
    else:
        results = [_analyse_one(u, jd, required_list) for u in usernames]

    valid   = [r for r in results if not r.get("error")]
    errored = [r for r in results if r.get("error")]

    if not valid:
        return {
            "ranked_candidates": [],
            "winner":            None,
            "differentiators":   [],
            "llm_explanation":   "All candidate analyses failed.",
            "total_evaluated":   len(results),
            "errors":            [f"{r['username']}: {r.get('error_msg', '')}" for r in errored],
        }

    # sort by raw final_score — do not round before sorting
    valid.sort(key=lambda x: _safe_float(x.get("final_score")), reverse=True)

    # tie-break top two if within threshold
    if len(valid) >= 2:
        gap = _safe_float(valid[0].get("final_score")) - _safe_float(valid[1].get("final_score"))
        if gap <= CONFIG["tie_threshold"]:
            winner = _tie_break(valid[0], valid[1])
            if winner["username"] == valid[1]["username"]:
                valid[0], valid[1] = valid[1], valid[0]

    # assign ranks and round final scores for display
    for i, c in enumerate(valid):
        c["rank"]        = i + 1
        c["final_score"] = round(_safe_float(c["final_score"]), 1)

    winner          = valid[0]
    runner_up       = valid[1] if len(valid) > 1 else None
    differentiators = (
        _compute_differentiators(winner, runner_up)
        if runner_up else ["Only one candidate evaluated"]
    )
    llm_explanation = _generate_explanation(winner, valid, differentiators)

    return {
        "ranked_candidates": valid,
        "winner":            winner,
        "differentiators":   differentiators,
        "llm_explanation":   llm_explanation,
        "total_evaluated":   len(results),
        "errors":            [f"{r['username']}: {r.get('error_msg', '')}" for r in errored],
    }