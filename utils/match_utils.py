from sentence_transformers import SentenceTransformer, util
import numpy as np

bert_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_best_candidates(job_desc_text, cvs):
    """Match job description against CVs and return top 5 candidates."""
    job_embedding = bert_model.encode(job_desc_text, convert_to_tensor=True)
    candidates = []
    
    for cv_name, cv_text in cvs.items():
        cv_embedding = bert_model.encode(cv_text, convert_to_tensor=True)
        score = util.pytorch_cos_sim(job_embedding, cv_embedding).item()
        candidates.append((cv_name, score))
    
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:5]
