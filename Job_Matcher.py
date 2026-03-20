import torch
from sentence_transformers import SentenceTransformer, util
import json
import re
import ollama

class TalentMatcher:
    def __init__(self):
        # Load the lightweight, high-performance vectorization model
        print("Loading vectorization model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.MATCH_THRESHOLD = 0.55 # Minimum cosine similarity to count as a match/adjacency

    def extract_requirements(self, job_description: str) -> list[str]:
        """
        approach 1: local LLM (phi3) for dynamic semantic extraction
        approach 2: huge Regex Taxonomy fallback IF the LLM fails or times out
        """
        print("Asking local phi3 to comprehend the job description...")
        
        # --- TIER 2 DATA: The Massive Fallback Taxonomy ---
        tech_taxonomy = [
            # Core Languages
            "Python", "Java", "C++", "C", "C#", "Rust", "Go", "JavaScript", "TypeScript", 
            "Ruby", "PHP", "Swift", "Kotlin", "Dart", "R", "Scala", "Lua", "Shell", "Bash",
            
            # Roles & Domains
            "Software Engineer", "Web Developer", "Frontend Developer", "Backend Developer", 
            "Full Stack Developer", "Data Scientist", "Data Engineer", "Machine Learning Engineer",
            "DevOps Engineer", "Cloud Architect", "Systems Administrator", "Product Manager",
            
            # AI, ML & Data
            "Machine Learning", "Artificial Intelligence", "AI", "Deep Learning", "Data Science", 
            "Data Processing", "Data Analytics", "NLP", "Natural Language Processing", 
            "Computer Vision", "LLM", "Generative AI", "Pandas", "NumPy", "TensorFlow", 
            "PyTorch", "Scikit-learn", "Big Data", "Hadoop", "Spark",
            
            # Web Frontend
            "React", "Vue", "Angular", "Svelte", "HTML", "CSS", "Tailwind", "Bootstrap", 
            "Web Development", "Frontend", "UI/UX", "Next.js", "WebAssembly",
            
            # Web Backend & Frameworks
            "Node.js", "Express", "Django", "Flask", "FastAPI", "Spring Boot", "ASP.NET", 
            "Ruby on Rails", "Laravel", "Backend", "REST API", "GraphQL", "Microservices",
            
            # Databases
            "Database Design", "SQL", "NoSQL", "MySQL", "PostgreSQL", "MongoDB", "Redis", 
            "Cassandra", "Oracle", "SQLite", "DynamoDB", "Elasticsearch", "Relational Databases",
            
            # Systems, Cloud & DevOps
            "Kernel", "Operating Systems", "Linux", "Unix", "Windows", "Embedded Systems",
            "Cloud", "Cloud Infrastructure", "AWS", "Azure", "GCP", "Google Cloud", 
            "Docker", "Kubernetes", "Containerization", "Terraform", "Ansible", "Jenkins", 
            "CI/CD", "DevOps", "GitHub Actions", "Serverless", "Networking",
            
            # Engineering Practices & Core Concepts
            "System Architecture", "Algorithms", "Data Structures", "Open Source", 
            "Open Source Contribution", "Git", "Version Control", "Agile", "Scrum", 
            "TDD", "Test Driven Development", "Security", "Cryptography"
        ]

        def regex_fallback(jd_text: str) -> list[str]:
            """Helper function to run the deterministic taxonomy scan"""
            print("LLM extraction failed or returned empty! Falling back to Regex Taxonomy scanner...")
            extracted = []
            jd_lower = jd_text.lower()
            for skill in tech_taxonomy:
                #to ensure only words are matched 
                pattern = r'\b' + re.escape(skill.lower()) + r'\b'
                if re.search(pattern, jd_lower):
                    extracted.append(skill)
            return extracted if extracted else ["Software Engineering", "Programming"]

        system_prompt = """
        You are a skill extraction engine. Extract all technical and soft skills, 
        roles, and engineering practices from the provided job description text. 
        Return ONLY a valid JSON array of strings. 
        Example: ["Python", "Machine Learning", "System Architecture", "Web Development"]
        """
        
        try:
            #calling local ollama instance('ollama run phi3' is active)
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
            
            # 1. Clean list returned
            if isinstance(extracted_data, list) and len(extracted_data) > 0:
                return extracted_data
                
            # 2. Weird dictionary returned (e.g., {"skills": ["Python"]}) for reviewing
            elif isinstance(extracted_data, dict):
                for key, value in extracted_data.items():
                    if isinstance(value, list) and len(value) > 0:
                        return value
            
            # 3. Valid JSON, but empty!, go back to approach 2
            return regex_fallback(job_description)
            
        except Exception as e:
            # Catch timeouts, connection errors, or JSON parsing failures, fall back to approach 2
            print(f"Ollama Error: {e}")
            return regex_fallback(job_description)

    def match_candidate(self, job_description: str, github_analysis: dict) -> dict:
        # 1. Get required skills from the JD
        required_skills = self.extract_requirements(job_description)
        
        # 2. Safely extract the candidate's verified skills from github analysis 
        candidate_skills_dict = github_analysis.get("skill_evidence", {}).get("skills", {})
        candidate_skills = list(candidate_skills_dict.keys())

        if not candidate_skills or not required_skills:
            return {
                "match_score": 0.0,
                "matched_skills": [],
                "missing_skills": required_skills,
                "reasoning": "Insufficient GitHub data to perform matching against requirements.",
                "error": True
            }

        # 3. Vectorize both sets of skills into dense embeddings
        req_embeddings = self.model.encode(required_skills, convert_to_tensor=True)
        cand_embeddings = self.model.encode(candidate_skills, convert_to_tensor=True)

        # 4. Compute cosine similarity matrix
        cosine_scores = util.cos_sim(req_embeddings, cand_embeddings)

        matched_skills = []
        missing_skills = []
        cumulative_score = 0.0

        # 5. Evaluate adjacencies and matches
        for i, req_skill in enumerate(required_skills):
            # Find the candidate's closest skill to this specific requirement
            best_match_idx = torch.argmax(cosine_scores[i]).item()
            best_score = cosine_scores[i][best_match_idx].item()
            best_cand_skill = candidate_skills[best_match_idx]

            if best_score >= self.MATCH_THRESHOLD:
                matched_skills.append({
                    "required_skill": req_skill,
                    "matched_evidence": best_cand_skill,
                    "adjacency_confidence": round(best_score * 100, 1)
                })
                cumulative_score += best_score
            else:
                missing_skills.append(req_skill)

        # 6. Calculate Final Scores and Generate Glass-Box Reasoning
        final_match_score = round((cumulative_score / len(required_skills)) * 100, 1)
        
        reasoning = f"Candidate satisfies {len(matched_skills)} of {len(required_skills)} core requirements based on GitHub evidence. "
        if missing_skills:
            reasoning += f"Notable skill gaps detected in: {', '.join(missing_skills)}."
        else:
            reasoning += "Candidate exhibits strong direct or adjacent capabilities for all role requirements."

        # Return the exact JSON structure required for the Glass-Box Recruiter impact area
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
    print("\n===== GLASS-BOX MATCH REPORT =====")
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