from pypdf import PdfReader

reader = PdfReader("suhc-heart-transplant-guide-2017-12-digital-version.pdf")
number_of_pages = len(reader.pages)
page = reader.pages[78]
text = page.extract_text()
print(f"Page {78 + 1} pdfplumber text: {text}")
print("------------------------------------------------------------------")

import pdfplumber

with pdfplumber.open('suhc-heart-transplant-guide-2017-12-digital-version.pdf') as pdf:
    page = pdf.pages[78]
    text = page.extract_text()
    print(f"Page {78 + 1} pdfplumber text: {text}")
    print("------------------------------------------------------------------")



import pymupdf  # PyMuPDF

doc = pymupdf.open('suhc-heart-transplant-guide-2017-12-digital-version.pdf')
page = doc.load_page(78)
text = page.get_text()
print(f"Page {78 + 1} fitz text: {text}")
print("------------------------------------------------------------------")



