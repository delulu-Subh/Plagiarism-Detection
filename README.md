# Plagiarism Detection System using TF-IDF and Cosine Similarity

## Overview

This project is a basic plagiarism detection system built with Python.

The system:

* Reads multiple `.txt` files
* Converts text into TF-IDF vectors
* Computes cosine similarity between documents
* Detects potentially plagiarized content
* Generates word clouds for visualization

The project demonstrates fundamental concepts from:

* Natural Language Processing (NLP)
* Machine Learning
* Text Vectorization
* Similarity Analysis

---

# Features

* Automatic scanning of `.txt` files
* TF-IDF vector generation
* Cosine similarity computation
* Pairwise document comparison
* Word cloud visualization
* Plagiarism threshold detection
* Memory-efficient sparse matrix handling

---

# Technologies Used

* Python
* Scikit-learn
* Matplotlib
* WordCloud

---

# Required Libraries

Install dependencies using:

```bash
pip install matplotlib scikit-learn wordcloud
```

---

# Project Workflow

```text
TXT Files
   ↓
Read Documents
   ↓
TF-IDF Vectorization
   ↓
Cosine Similarity Calculation
   ↓
Plagiarism Detection
   ↓
Word Cloud Visualization
```

---

# Code Explanation

# 1. Importing Libraries

```python
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
```

### Purpose

These libraries provide:

| Library             | Purpose                     |
| ------------------- | --------------------------- |
| `os`                | File and directory handling |
| `matplotlib`        | Visualization               |
| `TfidfVectorizer`   | Text vectorization          |
| `cosine_similarity` | Similarity measurement      |
| `WordCloud`         | Word cloud generation       |

---

# 2. Loading Text Files

```python
student_files = [
    file for file in os.listdir()
    if file.endswith(".txt")
]
```

### Purpose

* Scans the current directory
* Selects only `.txt` files

Example:

```text
student1.txt
student2.txt
notes.pdf
```

Result:

```python
["student1.txt", "student2.txt"]
```

---

## Reading File Contents

```python
student_docs = []

for file in student_files:
    with open(file, "r", encoding="utf-8", errors="ignore") as f:
        student_docs.append(f.read())
```

### Purpose

* Opens every text file
* Reads document content
* Stores content inside a list

---

# 3. Printing File Information

```python
print("\nLoaded Files:\n")
```

Prints heading.

---

```python
for filename, document in zip(student_files, student_docs):
```

### Purpose

Combines:

* filenames
* file contents

using `zip()`.

---

```python
print(f"Characters: {len(document)}")
```

Counts characters inside each document.

---

# 4. TF-IDF Vector Creation

## Function Definition

```python
def create_tfidf_vectors(docs):
```

Creates reusable TF-IDF vectorization function.

---

## Creating Vectorizer

```python
vectorizer = TfidfVectorizer(
    stop_words='english'
)
```

### TF-IDF Meaning

* TF = Term Frequency
* IDF = Inverse Document Frequency

### Purpose

* Important words receive higher weights
* Common words receive lower weights

---

## Stop Words

```python
stop_words='english'
```

Removes common words such as:

* the
* is
* and
* are

These words are not useful for plagiarism detection.

---

## Convert Documents to Vectors

```python
return vectorizer.fit_transform(docs)
```

### What Happens

| Method        | Purpose                    |
| ------------- | -------------------------- |
| `fit()`       | Learns vocabulary          |
| `transform()` | Converts text into vectors |

Combined:

```python
fit_transform()
```

---

## Sparse Matrix Optimization

```python
# Do NOT use .toarray()
```

### Reason

Using `.toarray()`:

* consumes large memory
* slows performance for large files

Sparse matrices store only non-zero values.

---

## Function Call

```python
doc_vectors = create_tfidf_vectors(student_docs)
```

Each document becomes a mathematical vector.

---

# 5. Plagiarism Detection

## Function Definition

```python
def find_plagiarism(files, vectors):
```

### Inputs

* filenames
* TF-IDF vectors

---

## Nested Loops

```python
for i in range(len(files)):
    for j in range(i + 1, len(files)):
```

### Purpose

Compares every file with every other file.

Avoids duplicate comparisons such as:

* A vs B
* B vs A

---

# Cosine Similarity

```python
similarity_score = cosine_similarity(
    vectors[i],
    vectors[j]
)[0][0]
```

### Purpose

Measures similarity between documents.

---

## Similarity Score Meaning

| Score | Meaning              |
| ----- | -------------------- |
| 0.0   | Completely different |
| 0.5   | Moderately similar   |
| 0.9   | Highly similar       |
| 1.0   | Identical            |

---

## Store Results

```python
results.append(
    (files[i], files[j], similarity_score)
)
```

Stores:

```python
("a.txt", "b.txt", 0.82)
```

---

# 6. Printing Results

```python
print(
    f"{file1} <-> {file2} "
    f"= Similarity: {score:.4f}"
)
```

### Purpose

Displays similarity scores.

Example:

```text
student1.txt <-> student2.txt = Similarity: 0.8231
```

---

# 7. Word Cloud Generation

## Function Definition

```python
def generate_word_cloud(document_text, filename):
```

Creates reusable visualization function.

---

## Creating Word Cloud

```python
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white'
).generate(document_text)
```

### Purpose

Generates visual representation of frequently used words.

---

## Displaying Word Cloud

```python
plt.figure(figsize=(10, 5))

plt.imshow(
    wordcloud,
    interpolation='bilinear'
)

plt.title(f"Word Cloud for {filename}")

plt.axis("off")

plt.show()
```

### Purpose

Displays the generated word cloud.

---

# 8. Detecting Highly Similar Files

## Similarity Threshold

```python
SIMILARITY_THRESHOLD = 0.50
```

### Meaning

If similarity score ≥ 50%:

* document pair is flagged as suspicious

---

## Threshold Check

```python
if score >= SIMILARITY_THRESHOLD:
```

Only highly similar files proceed.

---

# 9. Optional Single File Word Cloud

```python
target_document = "student2.txt"
```

Generates word cloud for a specific file.

---

## File Existence Check

```python
if target_document in student_files:
```

Prevents file-not-found errors.

---

# Machine Learning Concepts Used

## TF-IDF

TF-IDF helps identify important words in documents.

* Frequently occurring words inside one document gain importance
* Common words across all documents lose importance

---

## Cosine Similarity

Cosine similarity measures the angle between document vectors.

Formula:

```math
\cos(\theta)=\frac{A\cdot B}{|A||B|}
```

Where:

* `A · B` → dot product
* `|A|` → vector magnitude

### Interpretation

* Closer to `1` → more similar
* Closer to `0` → less similar

---

# Time Complexity

If there are `n` files:

```text
O(n²)
```

Because every file compares with every other file.

Examples:

| Files | Comparisons |
| ----- | ----------- |
| 10    | 45          |
| 100   | 4950        |

---

# Limitations

This approach:

* detects direct wording similarity
* cannot detect advanced paraphrasing
* ignores sentence structure
* may produce false positives

---

# Advanced Improvements

More advanced plagiarism systems use:

* BERT
* Sentence Transformers
* Semantic Similarity
* Embeddings
* Transformer Models

Possible upgrades:

* PDF/DOCX support
* GUI dashboard
* Database storage
* Web application deployment
* Semantic plagiarism detection

---

# Example Output

```text
Plagiarism Results:

student1.txt <-> student2.txt = Similarity: 0.8231

Possible plagiarism detected:
student1.txt <-> student2.txt
Score: 0.8231
```

---

# Conclusion

This project demonstrates how machine learning and NLP techniques can be applied to plagiarism detection using:

* TF-IDF vectorization
* Cosine similarity
* Text visualization

It serves as a strong beginner-to-intermediate NLP project for learning document similarity analysis.
