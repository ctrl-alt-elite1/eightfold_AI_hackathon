import torch
from sentence_transformers import SentenceTransformer, util
import json
import re

class TalentMatcher:
    def __init__(self):
        print("Loading vectorization model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.MATCH_THRESHOLD = 0.55 

    def extract_requirements(self, job_description: str) -> dict:
        """
        Tier 1: Local LLM (Phi-3) for weighted semantic extraction.
        Tier 2: Regex fallback.
        """
        print("Asking local phi3 to comprehend the job description...")
        
        tech_taxonomy = [
            "Python", "Java", "C++", "C", "C#", "Rust", "Go", "JavaScript", "TypeScript", 
            "Machine Learning", "Artificial Intelligence", "Data Science", "Computer Vision", 
            "React", "Vue", "Angular", "Node.js", "Django", "SQL", "MongoDB", "PostgreSQL",
            "Kernel", "Operating Systems", "Linux", "AWS", "Docker", "Kubernetes", "CI/CD",
            "System Architecture", "Algorithms", "Data Structures", "Open Source"
        ]

        def regex_fallback(jd_text: str) -> dict:
            print("⚠️ LLM extraction failed. Falling back to Regex Taxonomy...")
            extracted = []
            jd_lower = jd_text.lower()
            for skill in tech_taxonomy:
                pattern = r'\b' + re.escape(skill.lower()) + r'\b'
                if re.search(pattern, jd_lower):
                    extracted.append(skill)
            # If fallback triggers, assume all found skills are core
            return {"core_skills": extracted if extracted else ["Software Engineering"], "secondary_skills": []}

        # --- TIER 1: The Weighted & GitHub-Verifiable Prompt ---
        system_prompt = """
        You are an expert Technical AI Recruiter. Read the job description and extract ONLY skills that can be VERIFIED through a GitHub repository (e.g., Programming Languages, Frameworks, Architecture, Cloud Infrastructure, Database Design).
        
        CRITICAL RULES:
        1. EXCLUDE all non-code soft skills and methodologies (e.g., Agile, Scrum, Jira, Communication, Leadership, Pragmatic).
        2. Categorize the verified technical skills into two exact buckets:
           - "core_skills": The 3 to 6 absolute most critical foundational technologies required for the role.
           - "secondary_skills": Minor tools, libraries, edge-case technologies, or nice-to-haves.
        3. Extract atomic keywords (1 to 3 words max).
        4. You MUST return a JSON object with exactly these two keys.

        Example Output: 
        {
          "core_skills": ["Operating Systems", "C/C++", "Kernel Internals", "Systems Programming"],
          "secondary_skills": ["PyTorch", "GPUs", "Profiling Tools"]
        }
        """
        
        try:
            import ollama
            response = ollama.chat(
                model='phi3',
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': job_description}
                ],
                format='json', 
                options={'temperature': 0.1} 
            )
            
            raw_output = response['message']['content'].strip()
            extracted_data = json.loads(raw_output)
            
            # Safely extract the two tiers
            if isinstance(extracted_data, dict) and "core_skills" in extracted_data:
                return {
                    "core_skills": extracted_data.get("core_skills", []),
                    "secondary_skills": extracted_data.get("secondary_skills", [])
                }
            
            return regex_fallback(job_description)
            
        except Exception as e:
            print(f"Ollama Error: {e}")
            return regex_fallback(job_description)

    def match_candidate(self, job_description: str, github_analysis: dict) -> dict:
        requirements = self.extract_requirements(job_description)
        core_reqs = requirements.get("core_skills", [])
        sec_reqs = requirements.get("secondary_skills", [])
        all_reqs = core_reqs + sec_reqs

        candidate_skills_dict = github_analysis.get("skill_evidence", {}).get("skills", {})
        candidate_skills = list(candidate_skills_dict.keys())

        if not candidate_skills or not core_reqs:
            return {
                "match_score": 0.0,
                "matched_skills": [],
                "missing_skills": all_reqs,
                "reasoning": "Insufficient GitHub data to perform matching.",
                "error": True
            }

        cand_embeddings = self.model.encode(candidate_skills, convert_to_tensor=True)

        matched_skills = []
        missing_skills = []
        
        # Helper function to calculate vector matches for a specific tier
        def calculate_tier_score(req_list, tier_label):
            if not req_list: return 0.0
            req_embeddings = self.model.encode(req_list, convert_to_tensor=True)
            cosine_scores = util.cos_sim(req_embeddings, cand_embeddings)
            
            cumulative = 0.0
            for i, req_skill in enumerate(req_list):
                best_match_idx = torch.argmax(cosine_scores[i]).item()
                best_score = cosine_scores[i][best_match_idx].item()
                best_cand_skill = candidate_skills[best_match_idx]

                if best_score >= self.MATCH_THRESHOLD:
                    matched_skills.append({
                        "required_skill": f"{req_skill} ({tier_label})",
                        "matched_evidence": best_cand_skill,
                        "adjacency_confidence": round(best_score * 100, 1)
                    })
                    cumulative += best_score
                else:
                    missing_skills.append(f"{req_skill} ({tier_label})")
            
            return (cumulative / len(req_list)) * 100

        # Run the math for both tiers
        core_score = calculate_tier_score(core_reqs, "Core")
        sec_score = calculate_tier_score(sec_reqs, "Secondary")

        # --- THE WEIGHTED MATH ENGINE ---
        # Core is 80% of the grade, Secondary is 20%. 
        if sec_reqs:
            final_match_score = round((core_score * 0.80) + (sec_score * 0.20), 1)
        else:
            final_match_score = round(core_score, 1)

        # Generate Glass-Box Reasoning
        reasoning = f"Candidate scored {round(core_score, 1)}% on Core Technical Pillars and {round(sec_score, 1)}% on Secondary Tooling. "
        if core_score < 60:
            reasoning += "Candidate lacks critical foundational skills for this role."
        elif final_match_score > 75:
            reasoning += "Candidate is a highly aligned technical fit."

        return {
            "match_score": final_match_score,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "reasoning": reasoning,
            "error": False
        }

# --- Integration Example & Terminal Runner ---
if __name__ == "__main__":
    # Import your existing analysis function
    from Github_Analysis import analyze_github
    
    print("\nEightfold AI - Talent Intelligence Matcher\n")
    username = input("Enter GitHub username to evaluate: ").strip()
    if not username:
        print("Invalid username.")
        exit()
        
    raw_jd = input("Paste the Job Description: ").strip()
    if not raw_jd:
        print("Invalid Job Description.")
        exit()
    
    # Run GitHub extraction
    print("\nRunning GitHub Engineering Intelligence (Extracting Verified Signals)...")
    candidate_profile = analyze_github(username)
    
    # Run Semantic Matcher
    print("\nRunning Semantic Vector Matching Engine...")
    matcher = TalentMatcher()
    match_results = matcher.match_candidate(raw_jd, candidate_profile)
    
    # Output the Glass-Box Report
    print("\n===== EIGHTFOLD HACKATHON MATCH REPORT =====")
    print(f"Match Confidence:   {match_results.get('match_score')}%")
    print(f"Reasoning Engine:   {match_results.get('reasoning')}")
    
    print("\n--- Verified Matches & Adjacencies ---")
    if match_results.get('matched_skills'):
        for match in match_results.get('matched_skills', []):
            print(f"  ✓ {match['required_skill']} -> Backed by: {match['matched_evidence']} ({match['adjacency_confidence']}% confidence)")
    else:
        print("  (None found)")
    
    print("\n--- Missing Capabilities ---")
    if match_results.get('missing_skills'):
        for missing in match_results.get('missing_skills', []):
            print(f"  ✕ {missing}")
    else:
        print("  (None found)")
    print("\n==================================")