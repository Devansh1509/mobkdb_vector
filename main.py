import os
import monkdb
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

pdf_path = os.path.join("sample_docs", "sample.pdf")
print(f"📄 Reading PDF: {pdf_path}")

reader = PdfReader(pdf_path)
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

chunks = [text[i:i+500] for i in range(0, len(text), 500)]
print(f"✂️ Split into {len(chunks)} chunks. Inserting into MonkDB...")

db = monkdb.connect()
db.create_table("pdf_vectors", columns={
    "chunk": "TEXT",
    "embedding": "VECTOR(384)"
})

model = SentenceTransformer("all-MiniLM-L6-v2")

for chunk in chunks:
    emb = model.encode(chunk).tolist()
    db.insert("pdf_vectors", {"chunk": chunk, "embedding": emb})

query_text = "What is MonkDB used for?"
query_emb = model.encode(query_text).tolist()

results = db.query(f"SELECT chunk FROM pdf_vectors ORDER BY similarity(embedding, {query_emb}) DESC LIMIT 3")
for r in results:
    print("\n🔹 Relevant text snippet:\n", r["chunk"])

print("\n✅ PDF vector ingestion and semantic search completed successfully on 'pdf_vectors'.")
