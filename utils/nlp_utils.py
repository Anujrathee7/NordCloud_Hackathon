import spacy

nlp = spacy.load("en_core_web_sm")

def extract_keywords(text):
    """Extract important keywords from job description using NLP."""
    doc = nlp(text)
    keywords = set()

    # Extract named entities (Organizations, Skills, Positions, etc.)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PERSON", "GPE", "NORP", "PRODUCT", "WORK_OF_ART"]:
            keywords.add(ent.text)

    # Extract noun chunks (phrases containing skills/roles)
    for chunk in doc.noun_chunks:
        keywords.add(chunk.text.lower())

    return list(keywords)
