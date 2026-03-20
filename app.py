"""
app.py
------
Talent Intelligence System — Streamlit UI
Areas 01 + 02: Signal Extraction + Learning Trajectory

Run:
    streamlit run app.py

Required env vars (at least one):
    GITHUB_TOKEN      (optional but avoids rate limits)
    GROQ_API_KEY      (free — recommended for hackathon)
    OPENAI_API_KEY
    ANTHROPIC_API_KEY
"""

import streamlit as st
import os
import sys

# Make sure local modules are importable
sys.path.insert(0, os.path.dirname(__file__))

from github_analyser import analyse_github, GithubProfile
from skill_graph import build_skill_graph, compute_learning_trajectory
from llm_engine import extract_skills_from_text, analyse_gap, generate_match_report, bias_check

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
# Sidebar: API keys + settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")

    github_token = st.text_input(
        "GitHub Token (optional)", type="password",
        value=os.getenv("GITHUB_TOKEN", ""),
        help="Avoids GitHub rate limits. Generate at github.com/settings/tokens",
    )
    if github_token:
        os.environ["GITHUB_TOKEN"] = github_token

    llm_key_type = st.selectbox("LLM Provider", ["Groq (free)", "OpenAI", "Anthropic"])
    llm_key = st.text_input(f"{llm_key_type} API Key", type="password")
    if llm_key:
        key_map = {"Groq (free)": "GROQ_API_KEY", "OpenAI": "OPENAI_API_KEY", "Anthropic": "ANTHROPIC_API_KEY"}
        os.environ[key_map[llm_key_type]] = llm_key

    st.divider()
    st.caption("Built for Techkriti '26 Hackathon")

# ---------------------------------------------------------------------------
# Build skill graph once (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_skill_graph():
    return build_skill_graph()

skill_graph = get_skill_graph()

# ---------------------------------------------------------------------------
# Main layout: two columns
# ---------------------------------------------------------------------------

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("👤 Candidate Input")

    github_username = st.text_input(
        "GitHub Username",
        placeholder="e.g. torvalds",
        help="Public GitHub profile to analyse",
    )

    resume_text = st.text_area(
        "Resume / Additional Skills (optional)",
        height=160,
        placeholder="Paste resume text or a skill list here. This supplements GitHub signals.",
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
    if not github_username and not resume_text:
        st.error("Please enter a GitHub username or paste resume text.")
        st.stop()
    if not job_description:
        st.error("Please paste a job description.")
        st.stop()

    # -----------------------------------------------------------------------
    # Step 1: GitHub analysis (Area 01)
    # -----------------------------------------------------------------------
    github_profile = None
    github_skills = []

    if github_username:
        with st.spinner(f"Analysing GitHub profile: @{github_username}…"):
            try:
                github_profile = analyse_github(github_username)
                github_skills = github_profile.inferred_skills
                st.success(f"✓ GitHub profile loaded — {github_profile.total_repos} repos, "
                           f"{github_profile.total_stars} stars, "
                           f"complexity {github_profile.avg_repo_complexity}/10")
            except Exception as e:
                st.warning(f"GitHub error: {e}. Continuing with resume only.")

    # -----------------------------------------------------------------------
    # Step 2: Skill extraction from resume + GitHub
    # -----------------------------------------------------------------------
    all_candidate_skills = list(set(github_skills))

    if resume_text:
        with st.spinner("Extracting skills from resume…"):
            resume_skills = extract_skills_from_text(resume_text, context_hint="This is a resume.")
            all_candidate_skills = list(set(all_candidate_skills + resume_skills))

    if not all_candidate_skills:
        st.error("No skills could be extracted. Check your GitHub token or add resume text.")
        st.stop()

    # Extract required skills from JD
    with st.spinner("Parsing job description…"):
        jd_skills = extract_skills_from_text(job_description, context_hint="This is a job description.")

    # -----------------------------------------------------------------------
    # Step 3: LLM gap analysis (Area 01 — explainability)
    # -----------------------------------------------------------------------
    with st.spinner("Running gap analysis…"):
        gap = analyse_gap(
            candidate_skills=all_candidate_skills,
            job_description=job_description,
            github_summary=github_profile.raw_summary if github_profile else "",
        )

    # -----------------------------------------------------------------------
    # Step 4: Learning trajectory (Area 02)
    # -----------------------------------------------------------------------
    with st.spinner("Computing learning trajectory…"):
        trajectory = compute_learning_trajectory(
            candidate_skills=all_candidate_skills,
            required_skills=jd_skills,
            graph=skill_graph,
        )

    # -----------------------------------------------------------------------
    # Step 5: Explainability report
    # -----------------------------------------------------------------------
    with st.spinner("Generating explainability report…"):
        report = generate_match_report(
            username=github_username or "Candidate",
            candidate_skills=all_candidate_skills,
            gap_analysis=gap,
            trajectory_reasoning=trajectory.reasoning,
            job_title=job_title or "the role",
        )

    # -----------------------------------------------------------------------
    # Step 6: Bias check
    # -----------------------------------------------------------------------
    with st.spinner("Running bias check…"):
        bias = bias_check(all_candidate_skills, job_description)

    # ===================================================================
    # RESULTS DISPLAY
    # ===================================================================

    st.divider()
    st.header("📊 Results")

    # --- Top-line score row ---
    score_col, ttp_col, adj_col = st.columns(3)

    match_score = gap.get("match_score", 0)
    score_color = "green" if match_score >= 70 else "orange" if match_score >= 45 else "red"

    with score_col:
        st.metric("Match Score", f"{match_score}/100")
        st.progress(match_score / 100)

    with ttp_col:
        weeks = trajectory.total_weeks_to_productivity
        band = trajectory.productivity_band
        label = "Immediate" if weeks == 0 else f"~{weeks} weeks"
        st.metric("Time to Productivity", label)
        st.caption(f"Band: **{band}**")

    with adj_col:
        st.metric("Adjacency Score", f"{trajectory.adjacency_score}/100")
        st.caption("Combines current skills + learning potential")

    st.divider()

    # --- Two-column results ---
    res_left, res_right = st.columns([1, 1], gap="large")

    with res_left:
        # Skill signals
        st.subheader("🔬 Signal Extraction (Area 01)")

        if github_profile:
            with st.expander("GitHub Signals", expanded=True):
                st.write(f"**Top languages:** {', '.join(github_profile.top_languages[:5])}")
                st.write(f"**Repos:** {github_profile.total_repos}  |  "
                         f"**Stars:** {github_profile.total_stars}  |  "
                         f"**Complexity:** {github_profile.avg_repo_complexity}/10")
                st.caption(github_profile.raw_summary)

        matched = gap.get("matched_skills", [])
        missing = gap.get("missing_skills", [])
        transferable = gap.get("transferable_skills", [])

        if matched:
            st.success(f"**✓ Matched skills ({len(matched)}):** {', '.join(matched)}")
        if missing:
            st.error(f"**✗ Missing skills ({len(missing)}):** {', '.join(missing)}")
        if transferable:
            st.info(f"**↝ Transferable:** {', '.join(transferable)}")

        # Bias check
        st.subheader("⚖️ Bias Check")
        bc = bias
        if bc["bias_check_passed"]:
            st.success(bc["verdict"])
        else:
            st.warning(bc["verdict"])
        st.caption(
            f"Score with context: {bc['score_with_context']}  |  "
            f"Score without demographics: {bc['score_without_demographics']}  |  "
            f"Δ = {bc['delta']} pts"
        )

    with res_right:
        # Learning trajectory
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

        # Candidate's full skill set
        with st.expander("All detected candidate skills"):
            st.write(", ".join(sorted(all_candidate_skills)))

    # --- Explainability report ---
    st.divider()
    st.subheader("📝 Explainability Report")
    st.info(report)

    st.caption(
        "This system does not rely on name, gender, or institutional prestige. "
        "Scores are derived from verified skill signals and graph-based learning trajectory modelling."
    )
