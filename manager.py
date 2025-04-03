from flask import Flask, request, render_template, jsonify
import os
import fitz  # PyMuPDF
import docx
import spacy
from sentence_transformers import SentenceTransformer, util
import numpy as np

app = Flask(__name__)

# Load NLP model
nlp = spacy.load("en_core_web_sm")
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_text_from_file(file_path):
    """Extract text from PDF or Word document"""
    text = ""
    if file_path.endswith(".pdf"):
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text("text")
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
    return text

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

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        job_file = request.files["job_desc"]
        cv_files = request.files.getlist("cvs")
        
        if not job_file or not cv_files:
            return "Missing files", 400
        
        job_path = os.path.join(UPLOAD_FOLDER, job_file.filename)
        job_file.save(job_path)
        job_desc_text = extract_text_from_file(job_path)
        
        cvs = {}
        for cv in cv_files:
            cv_path = os.path.join(UPLOAD_FOLDER, cv.filename)
            cv.save(cv_path)
            cvs[cv.filename] = extract_text_from_file(cv_path)
        
        top_candidates = get_best_candidates(job_desc_text, cvs)
        return jsonify({"top_candidates": top_candidates})
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
