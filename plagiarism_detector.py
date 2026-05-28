import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud

# -----------------------------
# STEP 1: LOAD ALL TXT FILES
# -----------------------------

student_files = [
    file for file in os.listdir()
    if file.endswith(".txt")
]

student_docs = []

for file in student_files:
    with open(file, "r", encoding="utf-8", errors="ignore") as f:
        student_docs.append(f.read())

# Print loaded files
print("\nLoaded Files:\n")

for filename, document in zip(student_files, student_docs):
    print(f"File: {filename}")
    print(f"Characters: {len(document)}")
    print("-" * 40)

# -----------------------------
# STEP 2: CREATE TF-IDF VECTORS
# -----------------------------

def create_tfidf_vectors(docs):

    vectorizer = TfidfVectorizer(
        stop_words='english'
    )

    # IMPORTANT:
    # Do NOT use .toarray()
    # Keeps memory usage low for huge files
    return vectorizer.fit_transform(docs)

doc_vectors = create_tfidf_vectors(student_docs)

# -----------------------------
# STEP 3: PLAGIARISM DETECTION
# -----------------------------

def find_plagiarism(files, vectors):

    results = []

    for i in range(len(files)):

        for j in range(i + 1, len(files)):

            similarity_score = cosine_similarity(
                vectors[i],
                vectors[j]
            )[0][0]

            results.append(
                (files[i], files[j], similarity_score)
            )

    return results

plagiarism_results = find_plagiarism(
    student_files,
    doc_vectors
)

# -----------------------------
# STEP 4: PRINT RESULTS
# -----------------------------

print("\nPlagiarism Results:\n")

for file1, file2, score in plagiarism_results:

    print(
        f"{file1} <-> {file2} "
        f"= Similarity: {score:.4f}"
    )

# -----------------------------
# STEP 5: WORD CLOUD FUNCTION
# -----------------------------

def generate_word_cloud(document_text, filename):

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white'
    ).generate(document_text)

    plt.figure(figsize=(10, 5))

    plt.imshow(
        wordcloud,
        interpolation='bilinear'
    )

    plt.title(f"Word Cloud for {filename}")

    plt.axis("off")

    plt.show()

# -----------------------------
# STEP 6: GENERATE WORD CLOUDS
# FOR HIGHLY SIMILAR FILES
# -----------------------------

SIMILARITY_THRESHOLD = 0.50

for file1, file2, score in plagiarism_results:

    if score >= SIMILARITY_THRESHOLD:

        print(
            f"\nPossible plagiarism detected:"
            f"\n{file1} <-> {file2}"
            f"\nScore: {score:.4f}"
        )

        # Generate word cloud for file1
        with open(file1, "r", encoding="utf-8", errors="ignore") as f:
            generate_word_cloud(
                f.read(),
                file1
            )

        # Generate word cloud for file2
        with open(file2, "r", encoding="utf-8", errors="ignore") as f:
            generate_word_cloud(
                f.read(),
                file2
            )

# -----------------------------
# OPTIONAL:
# GENERATE WORD CLOUD FOR
# A SPECIFIC FILE
# -----------------------------
target_document = "student2.txt"
if target_document in student_files:
    with open(
        target_document,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:
        generate_word_cloud(
            f.read(),
            target_document
        )
else:
    print(f"\n{target_document} not found.")