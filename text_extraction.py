import PyPDF2

def extract_text_from_pdf(pdf_file):
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

pdf_file = 'description/JD1.pdf'
text = extract_text_from_pdf(pdf_file)
print(text)
