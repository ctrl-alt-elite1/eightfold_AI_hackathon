"""
app.py
------
Talent Intelligence System — Streamlit UI
Areas 01 + 02: Signal Extraction + Learning Trajectory

Run:
    streamlit run app.py

Required env vars:
    GITHUB_TOKEN   (optional but strongly recommended)
"""

import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from Github_Analysis import analyze_github
from Job_Matcher import TalentMatcher
from skill_graph import build_skill_graph, compute_learning_trajectory

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Talent Intelligence",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Talent Intelligence System")
st.caption("Post-Resume Era · Areas 01 & 02 · Eightfold AI × Techkriti '26")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")

    github_token = st.text_input(
        "GitHub Token (recommended)", type="password",
        value=os.getenv("GITHUB_TOKEN", ""),
        help="Avoids GitHub rate limits. Generate at github.com/settings/tokens",
    )
    if github_token:
        os.environ["GITHUB_TOKEN"] = github_token

    st.divider()
    st.caption("Built for Techkriti '26 Hackathon")

# ---------------------------------------------------------------------------
# Cached resources — built once per session
# ---------------------------------------------------------------------------

@st.cache_resource
def get_skill_graph():
    return build_skill_graph()

@st.cache_resource
def get_matcher():
    return TalentMatcher()   # loads sentence-transformer model once

skill_graph = get_skill_graph()
matcher     = get_matcher()

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("👤 Candidate Input")

    github_username = st.text_input(
        "GitHub Username",
        placeholder="e.g. torvalds",
        help="Public GitHub profile to analyse",
    )

with col_right:
    st.subheader("📋 Job Description")

    job_title = st.text_input("Job Title", placeholder="e.g. Senior ML Engineer")

    job_description = st.text_area(
        "Paste the full job description",
        height=200,
        placeholder="Include required skills, responsibilities, and tech stack.",
    )

# ---------------------------------------------------------------------------
# Analyse button
# ---------------------------------------------------------------------------

st.divider()
run_btn = st.button("🚀 Analyse Candidate", type="primary", use_container_width=True)

if run_btn:
    if not github_username:
        st.error("Please enter a GitHub username.")
        st.stop()
    if not job_description:
        st.error("Please paste a job description.")
        st.stop()

    # -----------------------------------------------------------------------
    # Step 1: GitHub analysis (Area 01)
    # -----------------------------------------------------------------------
    github_result = None

    with st.spinner(f"Analysing GitHub profile: @{github_username}…"):
        try:
            github_result = analyze_github(github_username)

            intel    = github_result["github_engineering_intelligence"]
            profile  = github_result["developer_profile"]
            behavior = github_result["engineering_behavior"]
            skills   = github_result["skill_evidence"]

            st.success(
                f"✓ @{github_username} loaded — "
                f"Score {intel['overall_score']}/100 · "
                f"{intel['confidence']} confidence · "
                f"Archetype: {profile['archetype']} · "
                f"Velocity: {profile['learning_velocity']}"
            )
        except Exception as e:
            st.error(f"GitHub analysis failed: {e}")
            st.stop()

    # -----------------------------------------------------------------------
    # Step 2: Semantic vector matching — JD vs GitHub skills (Area 01)
    # -----------------------------------------------------------------------
    with st.spinner("Running semantic vector matching…"):
        match = matcher.match_candidate(job_description, github_result)

    # Check if matching failed due to insufficient GitHub data
    if match.get("error"):
        st.error(
            "Insufficient GitHub data to perform matching. "
            "The profile may have no public repositories or the token may be missing."
        )
        st.stop()

    # -----------------------------------------------------------------------
    # Step 3: Learning trajectory (Area 02)
    # -----------------------------------------------------------------------

    # Flat skill list from GitHub domain skills for the graph
    candidate_skill_list = list(
        github_result["skill_evidence"].get("skills", {}).keys()
    )

    # Required skills = everything from matched + missing
    required_skill_list = [
        m["required_skill"] for m in match.get("matched_skills", [])
    ] + match.get("missing_skills", [])

    with st.spinner("Computing learning trajectory…"):
        trajectory = compute_learning_trajectory(
            candidate_skills=candidate_skill_list,
            required_skills=required_skill_list,
            graph=skill_graph,
        )

    # ===================================================================
    # RESULTS DISPLAY
    # ===================================================================

    st.divider()
    st.header("📊 Results")

    # --- Top-line score row ---
    score_col, ttp_col, adj_col, github_col = st.columns(4)

    match_score = match.get("match_score", 0)

    with score_col:
        st.metric("Match Score", f"{match_score}/100")
        st.progress(min(match_score / 100, 1.0))

    with ttp_col:
        weeks = trajectory.total_weeks_to_productivity
        band  = trajectory.productivity_band
        label = "Immediate" if weeks == 0 else f"~{weeks} weeks"
        st.metric("Time to Productivity", label)
        st.caption(f"Band: **{band}**")

    with adj_col:
        st.metric("Adjacency Score", f"{trajectory.adjacency_score}/100")
        st.caption("Current skills + learning potential")

    with github_col:
        st.metric("GitHub Score", f"{intel['overall_score']}/100")
        st.caption(f"{profile['archetype']} · {profile['learning_velocity']} velocity")

    st.divider()

    # --- Two-column results ---
    res_left, res_right = st.columns([1, 1], gap="large")

    with res_left:
        st.subheader("🔬 Signal Extraction (Area 01)")

        # GitHub behaviour signals
        with st.expander("GitHub Engineering Signals", expanded=True):
            b1, b2, b3 = st.columns(3)
            with b1:
                st.metric("Consistency", f"{behavior['consistency']:.0f}%")
            with b2:
                st.metric("Depth", f"{behavior['depth']:.0f}%")
            with b3:
                st.metric("Exploration", f"{behavior['exploration']:.0f}%")

            domain_skills = skills.get("skills", {})
            if domain_skills:
                st.write("**Verified domain skills:**")
                for domain, score in sorted(domain_skills.items(), key=lambda x: -x[1])[:6]:
                    badge = "🟢" if score > 70 else "🟡"
                    st.progress(score / 100, text=f"{badge} {domain} ({score:.0f})")

            for s in intel.get("strengths", []):
                st.success(f"✓ {s}")
            for r in intel.get("risks", []):
                st.warning(f"⚠ {r}")

        # Semantic match results
        matched_list = match.get("matched_skills", [])   # list of dicts
        missing_list = match.get("missing_skills", [])   # list of strings

        if matched_list:
            st.write(f"**✓ Matched capabilities ({len(matched_list)}):**")
            for m in matched_list:
                st.success(
                    f"✓ **{m['required_skill']}** → "
                    f"evidenced by *{m['matched_evidence']}* "
                    f"({m['adjacency_confidence']}% confidence)"
                )

        if missing_list:
            st.error(f"**✗ Missing ({len(missing_list)}):** {', '.join(missing_list)}")

        # Bias note
        st.subheader("⚖️ Bias Awareness")
        st.info(
            "Scores are derived purely from GitHub artefacts — commit patterns, "
            "language depth, and project complexity. Name, gender, university, "
            "and demographic fields are never used in this system."
        )

    with res_right:
        st.subheader("📈 Learning Trajectory (Area 02)")

        if trajectory.skill_gaps:
            for gap_item in trajectory.skill_gaps:
                if gap_item.distance == -1:
                    icon = "🔴"
                    desc = f"No adjacent skill — cold start (~{gap_item.weeks_to_acquire}w)"
                elif gap_item.distance == 1:
                    icon = "🟢"
                    desc = f"1 hop from **{gap_item.via}** (~{gap_item.weeks_to_acquire}w, high confidence)"
                elif gap_item.distance == 2:
                    icon = "🟡"
                    desc = f"2 hops via **{gap_item.via}** (~{gap_item.weeks_to_acquire}w, medium confidence)"
                else:
                    icon = "🟠"
                    desc = f"{gap_item.distance} hops via **{gap_item.via}** (~{gap_item.weeks_to_acquire}w)"

                st.markdown(f"{icon} **{gap_item.skill}** — {desc}")
        else:
            st.success("No skill gaps — candidate meets all requirements.")

        # Tech evolution
        evo = github_result.get("analytics", {}).get("tech_evolution", {})
        if evo:
            with st.expander("Tech evolution timeline"):
                for year in sorted(evo.keys())[-5:]:
                    st.write(f"**{year}:** {', '.join(evo[year])}")

        with st.expander("All detected candidate skills"):
            st.write(", ".join(sorted(candidate_skill_list)))

    # --- Explainability report ---
    st.divider()
    st.subheader("📝 Explainability Report")

    report_lines = [
        f"**Candidate:** @{github_username}  |  **Role:** {job_title or 'the role'}",
        f"**Match Score:** {match_score}/100  |  "
        f"**GitHub Engineering Score:** {intel['overall_score']}/100  |  "
        f"**Confidence:** {intel['confidence']}",
        "",
        match.get("reasoning", ""),
        "",
        f"**Developer Profile:** {profile['archetype']} — {profile['learning_velocity']} "
        f"learning velocity. Potential score: {profile['potential_score']:.0f}/100.",
        "",
        f"**Time to Productivity:** {label} ({band}). "
        f"Adjacency score: {trajectory.adjacency_score}/100.",
        "",
        f"**Strengths:** {', '.join(intel['strengths']) if intel['strengths'] else 'None detected.'}",
        f"**Risks:** {', '.join(intel['risks']) if intel['risks'] else 'None detected.'}",
    ]

    st.info("\n".join(report_lines))

    st.caption(
        "This system does not rely on name, gender, or institutional prestige. "
        "Scores are derived from verified GitHub signals and semantic vector matching."
    )
