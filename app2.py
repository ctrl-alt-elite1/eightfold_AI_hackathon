"""
app.py
------
Talent Intelligence System — Streamlit UI
Areas 01 + 02 + 04

Run:
    streamlit run app.py
"""

import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from Github_Analysis import analyze_github
from Job_Matcher import TalentMatcher
from skill_graph import build_skill_graph, compute_learning_trajectory
from hiring_intelligence import evaluate_candidates, run_bias_check

st.set_page_config(
    page_title="Talent Intelligence",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Talent Intelligence System")
st.caption("Evidence-Based Hiring Decisions · Eightfold AI × Techkriti '26")

# ---------------------------------------------------------------------------
# Hiring recommendation layer — rule-based, deterministic
# ---------------------------------------------------------------------------

def hiring_verdict(match_score: float, confidence: str, data_quality: str = "MEDIUM") -> tuple:
    """
    Returns (verdict_label, verdict_color, verdict_explanation).
    Pure rule-based logic — no LLM.
    """
    multiplier = {"HIGH": 1.0, "MEDIUM": 0.95, "LOW": 0.85}.get(confidence, 0.9)
    dq_penalty = {"HIGH": 0, "MEDIUM": 3, "LOW": 8}.get(data_quality, 3)
    adjusted   = (match_score * multiplier) - dq_penalty

    if adjusted >= 75:
        return "Strong Hire", "green", "Candidate demonstrates strong verified capability for this role."
    elif adjusted >= 60:
        return "Interview", "blue", "Candidate shows solid signals. Recommend a technical interview to confirm."
    elif adjusted >= 45:
        return "Consider", "orange", "Partial fit. Candidate may need upskilling in key areas."
    else:
        return "Weak Fit", "red", "Candidate does not meet core requirements based on available signals."

def confidence_narrative(confidence: str, data_quality: str, repo_count: int) -> str:
    """Plain English confidence explanation for recruiters."""
    base = {
        "HIGH":   "High confidence",
        "MEDIUM": "Moderate confidence",
        "LOW":    "Low confidence — limited data",
    }.get(confidence, "Moderate confidence")

    if repo_count >= 8:
        detail = "based on rich GitHub history with consistent contributions."
    elif repo_count >= 3:
        detail = "based on moderate GitHub activity. More data would improve accuracy."
    else:
        detail = "due to sparse GitHub data. Treat scores as indicative only."

    return f"{base} — {detail}"


def domain_distribution(skills: dict) -> dict:
    """Infer backend/frontend/ML/devops distribution from skill names."""
    domain_keywords = {
        "Backend":  ["backend", "python", "java", "go", "rust", "django", "fastapi", "flask", "node", "sql", "database", "api", "server"],
        "Frontend": ["frontend", "react", "vue", "angular", "javascript", "typescript", "html", "css", "ui", "web"],
        "ML/AI":    ["machine learning", "deep learning", "pytorch", "tensorflow", "data science", "nlp", "ai", "model"],
        "DevOps":   ["devops", "docker", "kubernetes", "ci/cd", "aws", "cloud", "linux", "shell", "terraform"],
    }
    dist = {d: 0 for d in domain_keywords}
    total_score = sum(skills.values()) or 1

    for skill_name, score in skills.items():
        skill_lower = skill_name.lower()
        for domain, keywords in domain_keywords.items():
            if any(kw in skill_lower for kw in keywords):
                dist[domain] += score
                break

    total = sum(dist.values()) or 1
    return {d: round((v / total) * 100, 1) for d, v in dist.items() if v > 0}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")
    github_token = st.text_input(
        "GitHub Token (recommended)", type="password",
        value=os.getenv("GITHUB_TOKEN", ""),
    )
    if github_token:
        os.environ["GITHUB_TOKEN"] = github_token
    st.divider()
    st.caption("Built for Techkriti '26 Hackathon")

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource
def get_skill_graph():
    return build_skill_graph()

@st.cache_resource
def get_matcher():
    return TalentMatcher()

skill_graph = get_skill_graph()
matcher     = get_matcher()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2 = st.tabs(["Single Candidate", "Compare Candidates"])

# ===========================================================================
# TAB 1 — Single Candidate
# ===========================================================================

with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("👤 Candidate")
        github_username = st.text_input(
            "GitHub Username", placeholder="e.g. torvalds", key="single_username"
        )

    with col_right:
        st.subheader("📋 Job Description")
        job_title = st.text_input("Job Title", placeholder="e.g. Senior ML Engineer", key="single_title")
        job_description = st.text_area(
            "Paste the full job description",
            height=180,
            placeholder="Include required skills, responsibilities, and tech stack.",
            key="single_jd",
        )

    st.divider()
    run_btn = st.button("🚀 Analyse Candidate", type="primary", use_container_width=True, key="single_run")

    if run_btn:
        if not github_username:
            st.error("Please enter a GitHub username.")
            st.stop()
        if not job_description:
            st.error("Please paste a job description.")
            st.stop()

        github_result = None
        with st.spinner(f"Extracting engineering signals from @{github_username}…"):
            try:
                github_result = analyze_github(github_username)
                intel    = github_result["github_engineering_intelligence"]
                profile  = github_result["developer_profile"]
                behavior = github_result["engineering_behavior"]
                skills_d = github_result["skill_evidence"]
            except Exception as e:
                st.error(f"GitHub analysis failed: {e}")
                st.stop()

        with st.spinner("Running semantic capability matching…"):
            match = matcher.match_candidate(job_description, github_result)

        if match.get("error"):
            st.error("Insufficient GitHub data to perform matching.")
            st.stop()

        candidate_skill_list = list(github_result["skill_evidence"].get("skills", {}).keys())
        required_skill_list  = [
            m["required_skill"] for m in match.get("matched_skills", [])
        ] + match.get("missing_skills", [])

        with st.spinner("Computing learning trajectory…"):
            trajectory = compute_learning_trajectory(
                candidate_skills=candidate_skill_list,
                required_skills=required_skill_list,
                graph=skill_graph,
            )

        with st.spinner("Running bias validation…"):
            bias = run_bias_check(candidate_skill_list, job_description)

        # ===== DECISION FIRST =====
        st.divider()
        match_score   = match.get("match_score", 0)
        confidence    = intel.get("confidence", "MEDIUM")
        repo_count    = sum(github_result.get("analytics", {}).get("activity_timeline", {}).values() or [0])
        verdict, color, verdict_desc = hiring_verdict(match_score, confidence)
        conf_text     = confidence_narrative(confidence, "HIGH" if repo_count >= 8 else "MEDIUM" if repo_count >= 3 else "LOW", repo_count)

        st.header("🎯 Hiring Decision")

        verdict_styles = {
            "green":  ("✅", "success"),
            "blue":   ("🔵", "info"),
            "orange": ("⚠️", "warning"),
            "red":    ("❌", "error"),
        }
        icon, style = verdict_styles.get(color, ("🔵", "info"))
        getattr(st, style)(f"{icon} **{verdict}** — @{github_username}  |  Match: {match_score}/100  |  {verdict_desc}")

        # confidence + bias together as trust signals
        trust_col1, trust_col2 = st.columns(2)
        with trust_col1:
            st.caption(f"**Decision Confidence:** {conf_text}")
        with trust_col2:
            if bias["bias_check_passed"]:
                st.caption(f"**Bias Validation:** Score stable (Δ {bias['delta']} pts after removing demographic context) ✓")
            else:
                st.caption(f"**Bias Warning:** Score shifted {bias['delta']} pts — review JD language.")

        st.divider()

        # Supporting metrics
        st.subheader("📊 Supporting Evidence")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Capability Match", f"{match_score}/100")
            st.progress(min(match_score / 100, 1.0))
        with c2:
            weeks = trajectory.total_weeks_to_productivity
            label = "Immediate" if weeks == 0 else f"~{weeks} weeks"
            st.metric("Time to Productivity", label)
            st.caption(f"Ramp-up band: {trajectory.productivity_band}")
        with c3:
            st.metric("Learning Adjacency", f"{trajectory.adjacency_score}/100")
            st.caption("Transferable skill proximity")
        with c4:
            st.metric("Engineering Score", f"{intel['overall_score']}/100")
            st.caption(f"{profile['archetype']} · {profile['learning_velocity']} velocity")

        st.divider()
        res_left, res_right = st.columns([1, 1], gap="large")

        with res_left:
            st.subheader("🔬 Candidate Capability")

            # Domain distribution
            domain_dist = domain_distribution(skills_d.get("skills", {}))
            if domain_dist:
                st.write("**Engineering domain distribution:**")
                for domain, pct in sorted(domain_dist.items(), key=lambda x: -x[1]):
                    st.progress(pct / 100, text=f"{domain}: {pct}%")

            with st.expander("Verified skill signals", expanded=True):
                b1, b2, b3 = st.columns(3)
                with b1: st.metric("Consistency", f"{behavior['consistency']:.0f}%")
                with b2: st.metric("Depth", f"{behavior['depth']:.0f}%")
                with b3: st.metric("Exploration", f"{behavior['exploration']:.0f}%")

                for s in intel.get("strengths", []):
                    st.success(f"✓ {s}")
                for r in intel.get("risks", []):
                    st.warning(f"⚠ {r}")

            matched_list = match.get("matched_skills", [])
            missing_list = match.get("missing_skills", [])
            if matched_list:
                st.write(f"**✓ Verified capabilities ({len(matched_list)}):**")
                for m in matched_list:
                    st.success(
                        f"✓ **{m['required_skill']}** → {m['matched_evidence']} "
                        f"({m['adjacency_confidence']}% confidence)"
                    )
            if missing_list:
                st.error(f"**Capability gaps ({len(missing_list)}):** {', '.join(missing_list)}")

        with res_right:
            st.subheader("📈 Onboarding & Ramp-Up")
            st.caption(
                f"Estimated time to full productivity: **{label}** ({trajectory.productivity_band}). "
                f"Adjacency score {trajectory.adjacency_score}/100 indicates candidate's transferable skill proximity."
            )

            if trajectory.skill_gaps:
                for g in trajectory.skill_gaps:
                    if g.distance == -1:
                        icon = "🔴"; desc = f"No adjacent skill — cold start (~{g.weeks_to_acquire}w onboarding)"
                    elif g.distance == 1:
                        icon = "🟢"; desc = f"1 hop from **{g.via}** — fast transfer (~{g.weeks_to_acquire}w)"
                    elif g.distance == 2:
                        icon = "🟡"; desc = f"2 hops via **{g.via}** — moderate ramp (~{g.weeks_to_acquire}w)"
                    else:
                        icon = "🟠"; desc = f"{g.distance} hops via **{g.via}** (~{g.weeks_to_acquire}w)"
                    st.markdown(f"{icon} **{g.skill}** — {desc}")
            else:
                st.success("Candidate meets all skill requirements. Immediate productivity expected.")

            evo = github_result.get("analytics", {}).get("tech_evolution", {})
            if evo:
                with st.expander("Technology adoption timeline"):
                    for year in sorted(evo.keys())[-5:]:
                        st.write(f"**{year}:** {', '.join(evo[year])}")

            with st.expander("All detected capabilities"):
                st.write(", ".join(sorted(candidate_skill_list)))

        st.divider()
        st.subheader("📝 Decision Justification")
        weeks_label = "Immediate" if trajectory.total_weeks_to_productivity == 0 else f"~{trajectory.total_weeks_to_productivity} weeks"
        report_lines = [
            f"**Candidate:** @{github_username}  |  **Role:** {job_title or 'the role'}  |  **Verdict: {verdict}**",
            "",
            f"{verdict_desc} Match score of {match_score}/100 with {confidence.lower()} confidence.",
            "",
            match.get("reasoning", ""),
            "",
            f"**Onboarding outlook:** {weeks_label} to productivity ({trajectory.productivity_band} ramp-up band). "
            f"Adjacency score {trajectory.adjacency_score}/100 — candidate has transferable skills covering most gaps.",
            "",
            f"**Developer profile:** {profile['archetype']} archetype with {profile['learning_velocity'].lower()} learning velocity.",
            f"**Strengths:** {', '.join(intel['strengths']) if intel['strengths'] else 'None detected.'}",
            f"**Risks to watch:** {', '.join(intel['risks']) if intel['risks'] else 'None detected.'}",
            "",
            f"**Bias validation:** Score Δ = {bias['delta']} pts after demographic context removal. "
            f"{'Decision is demographically neutral.' if bias['bias_check_passed'] else 'Review JD for potential bias.'}",
        ]
        st.info("\n".join(report_lines))
        st.caption(
            "This system scores candidates purely on GitHub engineering signals. "
            "Name, gender, university, and demographic fields are never used."
        )


# ===========================================================================
# TAB 2 — Compare Candidates
# ===========================================================================

with tab2:
    st.subheader("👥 Multi-Candidate Hiring Decision")
    st.caption("Rank multiple candidates and surface the strongest hire with full evidence.")

    comp_left, comp_right = st.columns([1, 1], gap="large")

    with comp_left:
        usernames_input = st.text_area(
            "GitHub Usernames (one per line or comma-separated)",
            height=120,
            placeholder="torvalds\ngvanrossum\ndhh",
            key="compare_usernames",
        )

    with comp_right:
        comp_job_title = st.text_input("Job Title", placeholder="e.g. Backend Engineer", key="comp_title")
        comp_jd = st.text_area(
            "Job Description",
            height=180,
            placeholder="Paste the full job description here.",
            key="comp_jd",
        )

    parallel_mode = st.checkbox("Parallel analysis (faster)", value=False)
    st.divider()
    compare_btn = st.button("🏆 Find Best Candidate", type="primary", use_container_width=True, key="compare_run")

    if compare_btn:
        if not usernames_input.strip():
            st.error("Please enter at least one GitHub username.")
            st.stop()
        if not comp_jd.strip():
            st.error("Please paste a job description.")
            st.stop()

        raw = usernames_input.replace(",", "\n")
        usernames = [u.strip() for u in raw.splitlines() if u.strip()]

        if len(usernames) < 2:
            st.warning("Enter at least 2 usernames to compare.")
            st.stop()

        with st.spinner(f"Evaluating {len(usernames)} candidates…"):
            report = evaluate_candidates(
                candidate_usernames=usernames,
                job_description=comp_jd,
                parallel=parallel_mode,
            )

        ranked      = report.get("ranked_candidates", [])
        winner      = report.get("winner")
        diffs       = report.get("differentiators", [])
        explanation = report.get("llm_explanation", "")
        errors      = [e for e in report.get("errors", []) if e]

        if not ranked:
            st.error("No candidates could be analysed.")
            st.stop()

        if errors:
            with st.expander(f"⚠️ {len(errors)} candidate(s) failed"):
                for err in errors:
                    st.warning(err)

        # ===== DECISION FIRST =====
        st.divider()
        st.header("🎯 Hiring Decision")

        if winner:
            w_verdict, w_color, w_desc = hiring_verdict(
                winner["final_score"],
                winner.get("confidence", "MEDIUM"),
                winner.get("data_quality_flag", "MEDIUM"),
            )
            verdict_styles = {
                "green":  ("✅", "success"),
                "blue":   ("🔵", "info"),
                "orange": ("⚠️", "warning"),
                "red":    ("❌", "error"),
            }
            w_icon, w_style = verdict_styles.get(w_color, ("🔵", "info"))
            getattr(st, w_style)(
                f"{w_icon} **Recommended Hire: @{winner['username']}**  |  "
                f"Score: {winner['final_score']}/100  |  {w_desc}"
            )

            runner_up = ranked[1] if len(ranked) >= 2 else None
            if runner_up:
                st.caption(f"Runner-up: @{runner_up['username']} — {runner_up['final_score']}/100")

            # confidence narrative + bias together
            repo_count = 0
            w_conf = winner.get("confidence", "MEDIUM")
            conf_text = confidence_narrative(w_conf, winner.get("data_quality_flag", "MEDIUM"), repo_count)
            st.caption(f"**Decision confidence:** {conf_text}")

        st.divider()

        # Rankings table
        st.subheader("📊 Full Candidate Rankings")
        table_data = []
        for c in ranked:
            c_verdict, _, _ = hiring_verdict(c["final_score"], c.get("confidence", "MEDIUM"))
            table_data.append({
                "Rank":           c["rank"],
                "Username":       f"@{c['username']}",
                "Verdict":        c_verdict,
                "Final Score":    c["final_score"],
                "Capability Match": round(c.get("match_score", 0), 1),
                "Engineering":    round(c.get("engineering_score", 0), 1),
                "Learning Traj.": round(c.get("adjacency_score", 0), 1),
                "Confidence":     c.get("confidence", ""),
                "Archetype":      c.get("archetype", ""),
                "Data Quality":   c.get("data_quality_flag", ""),
            })
        st.dataframe(table_data, use_container_width=True)

        st.divider()

        # Why winner was selected
        st.subheader("🔍 Why This Candidate Was Selected")
        if winner and len(ranked) >= 2:
            runner_up = ranked[1]
            diff_col1, diff_col2 = st.columns(2)
            with diff_col1:
                st.write(f"**Winner: @{winner['username']}**")
                for d in diffs:
                    st.info(d)
            with diff_col2:
                st.write(f"**Runner-up: @{runner_up['username']}**")
                st.caption(
                    f"Score: {runner_up['final_score']}/100 · "
                    f"{runner_up.get('archetype')} · "
                    f"{runner_up.get('confidence')} confidence"
                )
                ru_verdict, _, ru_desc = hiring_verdict(
                    runner_up["final_score"], runner_up.get("confidence", "MEDIUM")
                )
                st.caption(f"Verdict: {ru_verdict} — {ru_desc}")

        st.divider()

        # Domain distribution per candidate
        st.subheader("🗺️ Engineering Domain Distribution")
        domain_cols = st.columns(min(len(ranked), 3))
        for i, c in enumerate(ranked[:3]):
            with domain_cols[i]:
                st.write(f"**@{c['username']}**")
                cand_skills = {s: 80.0 for s in c.get("matched_skills", [])}
                dist = domain_distribution(cand_skills)
                if dist:
                    for domain, pct in sorted(dist.items(), key=lambda x: -x[1]):
                        st.progress(pct / 100, text=f"{domain}: {pct}%")
                else:
                    st.caption("Domain distribution unavailable")

        st.divider()

        # Score breakdown
        st.subheader("📈 Score Breakdown")
        breakdown_cols = st.columns(min(len(ranked), 3))
        for i, c in enumerate(ranked[:3]):
            with breakdown_cols[i]:
                st.write(f"**#{c['rank']} @{c['username']}**")
                bd = c.get("score_breakdown", {})
                for component, val in bd.items():
                    st.progress(min(val / 35, 1.0), text=f"{component.capitalize()}: {val}")

        st.divider()

        # Onboarding comparison
        st.subheader("⏱️ Onboarding & Ramp-Up Comparison")
        traj_cols = st.columns(min(len(ranked), 3))
        for i, c in enumerate(ranked[:3]):
            with traj_cols[i]:
                st.write(f"**@{c['username']}**")
                traj = c.get("trajectory")
                if traj and traj.skill_gaps:
                    for g in traj.skill_gaps[:4]:
                        icon = "🟢" if g.distance == 1 else "🟡" if g.distance == 2 else "🔴"
                        st.markdown(f"{icon} {g.skill} (~{g.weeks_to_acquire}w)")
                    st.caption(
                        f"Time to productivity: {traj.total_weeks_to_productivity}w "
                        f"({traj.productivity_band})"
                    )
                elif traj:
                    st.success("No gaps — immediate productivity")

        st.divider()

        # LLM explanation
        st.subheader("📝 Hiring Intelligence Report")
        st.info(explanation)

        # Bias check on winner
        if winner:
            st.subheader("⚖️ Bias Validation")
            with st.spinner("Validating decision neutrality…"):
                winner_skills = winner.get("matched_skills", []) + winner.get("missing_skills", [])
                bias = run_bias_check(winner_skills, comp_jd)
            if bias["bias_check_passed"]:
                st.success(
                    f"✓ Decision is demographically neutral — "
                    f"score stable with Δ {bias['delta']} pts after removing demographic context."
                )
            else:
                st.warning(
                    f"Score shifted {bias['delta']} pts after removing demographic keywords. "
                    f"Review JD language for potential bias."
                )
            st.caption(
                f"Score with full context: {bias['score_with_context']} | "
                f"Score without demographics: {bias['score_without_demographics']}"
            )

        st.caption(
            "All rankings are fully deterministic. LLM was used only for the explanation text above. "
            "Scores are based solely on verified GitHub engineering signals."
        )
