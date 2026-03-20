"""
app.py
------
Talent Intelligence System — Streamlit UI
Areas 01 + 02: Signal Extraction + Learning Trajectory
"""

import streamlit as st
import os
import sys
import re

# Ensure local imports work
sys.path.insert(0, os.path.dirname(__file__))

from Github_Analysis import analyze_github
from Job_Matcher import TalentMatcher
from skill_graph import build_skill_graph, compute_learning_trajectory, SemanticSkillMapper

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
# Cached resources — Loads once per session to save memory and time
# ---------------------------------------------------------------------------
@st.cache_resource
def get_graph_infrastructure():
    """Initializes the Knowledge Graph and the Semantic Vector Mapper."""
    graph = build_skill_graph()
    # This loads the sentence-transformer model for the graph 'snapping'
    mapper = SemanticSkillMapper(graph)
    return graph, mapper

@st.cache_resource
def get_matcher():
    """Initializes the Job Matcher LLM and Vector Engine."""
    return TalentMatcher()

# Initialize infrastructure
skill_graph, mapper = get_graph_infrastructure()
matcher = get_matcher()

# ---------------------------------------------------------------------------
# Sidebar & Inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    github_token = st.text_input(
        "GitHub Token (recommended)", type="password",
        value=os.getenv("GITHUB_TOKEN", ""),
        help="Avoids GitHub rate limits.",
    )
    if github_token:
        os.environ["GITHUB_TOKEN"] = github_token
    st.divider()
    st.caption("Built for Techkriti '26 Hackathon")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("👤 Candidate Input")
    github_username = st.text_input("GitHub Username", placeholder="e.g. torvalds")

with col_right:
    st.subheader("📋 Job Description")
    job_title = st.text_input("Job Title", placeholder="e.g. Senior Systems Engineer")
    job_description = st.text_area("Paste the full job description", height=200)

st.divider()
run_btn = st.button("🚀 Analyse Candidate", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Execution Logic
# ---------------------------------------------------------------------------
if run_btn:
    if not github_username or not job_description:
        st.error("Please enter both a GitHub username and a job description.")
        st.stop()

    # Step 1: GitHub analysis (Area 01)
    with st.spinner(f"Analysing GitHub profile: @{github_username}…"):
        try:
            github_result = analyze_github(github_username)
            intel = github_result["github_engineering_intelligence"]
            profile = github_result["developer_profile"]
            behavior = github_result["engineering_behavior"]
            skills = github_result["skill_evidence"]
            st.success(f"✓ @{github_username} loaded — Score {intel['overall_score']}/100")
        except Exception as e:
            st.error(f"GitHub analysis failed: {e}")
            st.stop()

    # Step 2: Semantic vector matching (Area 01)
    with st.spinner("Running weighted semantic matching…"):
        match = matcher.match_candidate(job_description, github_result)
        if match.get("error"):
            st.error("Insufficient GitHub data to perform matching.")
            st.stop()

    # Step 3: Learning trajectory (Area 02 - Hybrid Mode)
    candidate_skill_list = list(github_result["skill_evidence"].get("skills", {}).keys())
    required_skill_list = [m["required_skill"] for m in match.get("matched_skills", [])] + match.get("missing_skills", [])

    with st.spinner("Computing hybrid learning trajectory…"):
        # FIXED: Passing 'mapper' as the required positional argument
        trajectory = compute_learning_trajectory(
            candidate_skills=candidate_skill_list,
            required_skills=required_skill_list,
            graph=skill_graph,
            mapper=mapper
        )

    # ---------------------------------------------------------------------------
    # Results Display
    # ---------------------------------------------------------------------------
    st.divider()
    st.header("📊 Intelligence Report")

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Match Score", f"{match.get('match_score', 0)}/100")
    with m2: st.metric("Time to Productivity", "Immediate" if trajectory.weeks == 0 else f"~{trajectory.weeks}w")
    with m3: st.metric("Adjacency Score", f"{trajectory.score}/100")
    with m4: st.metric("Archetype", profile['archetype'])

    res_left, res_right = st.columns([1, 1], gap="large")

    with res_left:
        st.subheader("🔬 Signal Extraction (Area 01)")
        with st.expander("GitHub Engineering Signals", expanded=True):
            b1, b2, b3 = st.columns(3)
            b1.metric("Consistency", f"{behavior['consistency']:.0f}%")
            b2.metric("Depth", f"{behavior['depth']:.0f}%")
            b3.metric("Exploration", f"{behavior['exploration']:.0f}%")

            domain_skills = skills.get("skills", {})
            for domain, score in sorted(domain_skills.items(), key=lambda x: -x[1])[:5]:
                st.progress(score / 100, text=f"{domain} ({score:.0f})")

        for m in match.get("matched_skills", []):
            st.success(f"✓ **{m['required_skill']}** matched via *{m['matched_evidence']}*")
        
        missing = match.get("missing_skills", [])
        if missing: st.error(f"**✗ Missing:** {', '.join(missing)}")

    with res_right:
        st.subheader("📈 Learning Trajectory (Area 02)")
        # FIXED: Updated loop for new Hybrid Trajectory data structure
        if trajectory.gaps:
            for gap in trajectory.gaps:
                if gap.distance == -1:
                    st.markdown(f"🔴 **{gap.original_skill}** — Cold start (~{gap.weeks}w)")
                elif gap.distance == 1:
                    st.markdown(f"🟢 **{gap.original_skill}** — Direct adjacency from **{gap.via}** (~{gap.weeks}w)")
                elif gap.distance <= 3:
                    st.markdown(f"🟡 **{gap.original_skill}** — Bridge via **{gap.via}** (~{gap.weeks}w)")
                else:
                    st.markdown(f"🟠 **{gap.original_skill}** — Broad transition via **{gap.via}** (~{gap.weeks}w)")
        else:
            st.success("No skill gaps detected.")

    st.divider()
    st.subheader("📝 Glass-Box Reasoning")
    st.info(f"{match.get('reasoning', '')}\n\n{trajectory.reasoning}")