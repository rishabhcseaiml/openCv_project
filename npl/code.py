# ============================
# NLP Text Preprocessing using NLTK
# ============================

# Step 1: Install NLTK (Run once in terminal)
# pip install nltk

# Step 2: Download Required Resources (Run once)
import nltk
print("Rishabh Raikwar uid 23bai70136")

# nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

# ============================
# Step 3: Input Text
# ============================

text = "Rishabh Raikwar study in chandigarh university as a be cse aiml student"

print("="*50)
print("Original Text")
print("="*50)
print(text)

# ============================
# 1. Sentence Segmentation
# ============================

from nltk.tokenize import sent_tokenize

sentences = sent_tokenize(text)

print("\n1. Sentence Segmentation")
print(sentences)

# ============================
# 2. Tokenization
# ============================

from nltk.tokenize import word_tokenize

tokens = word_tokenize(text)

print("\n2. Tokens")
print(tokens)

# ============================
# 3. Normalization
# ============================

normalized_tokens = [word.lower() for word in tokens]

print("\n3. Normalized Tokens")
print(normalized_tokens)

# ============================
# 4. Stop Word Removal
# ============================

from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

filtered_words = [
    word for word in normalized_tokens
    if word not in stop_words
]

print("\n4. After Stop Word Removal")
print(filtered_words)

# ============================
# 5. Frequency Analysis
# ============================

from nltk import FreqDist

frequency = FreqDist(filtered_words)

print("\n5. Frequency Distribution")
print(frequency)

print("\nMost Common Words")
print(frequency.most_common())

# ============================
# 6. Stemming
# ============================

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

stems = [stemmer.stem(word) for word in filtered_words]

print("\n6. Stemming")
print(stems)

# ============================
# 7. Lemmatization
# ============================

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

lemmas = [lemmatizer.lemmatize(word) for word in filtered_words]

print("\n7. Lemmatization")
print(lemmas)

# ============================
# 8. POS Tagging
# ============================

from nltk import pos_tag

pos_tags = pos_tag(tokens)

print("\n8. POS Tagging")
print(pos_tags)

# ============================
# 9. Named Entity Recognition
# ============================

from nltk import ne_chunk

named_entities = ne_chunk(pos_tags)

print("\n9. Named Entity Recognition")
print(named_entities)

# ============================
# End of Program
# ============================





!pip install lemminflect
import spacy
from lemminflect import getInflection, getAllInflections
nlp = spacy.load("en_core_web_sm")
sentence = "Children were playing happily."
doc = nlp(sentence)
for token in doc:
    print("-------------------------")
    print("Word       :", token.text)
    print("Lemma      :", token.lemma_)
    print("POS        :", token.pos_)
    print("Fine Tag   :", token.tag_)   # Penn Treebank tag
    print("Morphology :", token.morph)
    print(getInflection("play", tag="VBD"))   # past tense
print(getInflection("play", tag="VBG"))   # present participle
print(getInflection("child", tag="NNS"))  # plural noun
print(getInflection("run", tag="VBD"))    # irregular verb
forms = getAllInflections("write")
for tag, value in forms.items():
    print(tag, "=", value)
    word = "run"

print("Plural             :", getInflection(word, "NNS"))
print("Past               :", getInflection(word, "VBD"))
print("Present Participle :", getInflection(word, "VBG"))
print("Third Person       :", getInflection(word, "VBZ"))
print("Comparative        :", getInflection(word, "JJR"))
print("Superlative        :", getInflection(word, "JJS"))



# ===========================================
# Experiment 3: Regular Expressions in Python
# ===========================================

# Step 1: Import the Regular Expression module
import re

# Step 2: Sample text
print("Rishabh Raikwar 23bai70136")
text = "Contact us at support@example.com or sales@company.org. Call 9876543210 or 9123456789."

print("Original Text:")
print(text)

# ===========================================
# re.search()
# Finds the first occurrence anywhere
# ===========================================

print("\n----- re.search() -----")

m = re.search(r'\d{10}', text)

print("Match Object:", m)

if m:
    print("First Phone Number:", m.group())
else:
    print("No Match Found")

# ===========================================
# re.match()
# Checks only from the beginning
# ===========================================

print("\n----- re.match() -----")

m2 = re.match(r'Contact', text)

if m2:
    print("Matched Word:", m2.group())
else:
    print("No Match")

# Digits are not at the beginning
m3 = re.match(r'\d{10}', text)

print("Digit Match:", m3)

# ===========================================
# re.findall()
# Finds all matching patterns
# ===========================================

print("\n----- re.findall() -----")

# Find all email addresses
emails = re.findall(r'[\w.]+@[\w.]+\.\w+', text)

print("Emails Found:")
print(emails)

# Find all phone numbers
numbers = re.findall(r'\d{10}', text)

print("Phone Numbers:")
print(numbers)


# ==========================================================
# Experiment: Morphological Analysis using Python (NLTK)
# Aim:
# Analyze words by identifying prefixes, stems, suffixes,
# stemming, and lemmatization.
# ==========================================================

# Install NLTK (Run only once in Colab)
# !pip install nltk

import nltk
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer

# Download required resources (Run once)
nltk.download('wordnet')
nltk.download('omw-1.4')

print("Rishabh Raikwar 23BAI70136")

# ----------------------------------------------------------
# Sample Dataset
# ----------------------------------------------------------
words = [
    "unhappy",
    "replayed",
    "kindness",
    "studies",
    "running",
    "nationalization",
    "children",
    "playing",
    "redo",
    "disagree",
    "careless",
    "beautifully"
]

# ----------------------------------------------------------
# Prefixes and Suffixes (Simple Rule-Based)
# ----------------------------------------------------------
prefixes = [
    "un", "re", "dis", "pre", "mis", "non",
    "in", "im", "ir", "il", "over", "under"
]

suffixes = [
    "ing", "ed", "ly", "ness", "tion",
    "s", "es", "ment", "able", "ful",
    "less", "ize", "ization"
]

# ----------------------------------------------------------
# Initialize Stemmers and Lemmatizer
# ----------------------------------------------------------
porter = PorterStemmer()
snowball = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

# ----------------------------------------------------------
# Function for Morphological Analysis
# ----------------------------------------------------------
def analyze_word(word):
    prefix = "-"
    suffix = "-"
    stem = word

    # Detect Prefix
    for p in prefixes:
        if word.startswith(p) and len(word) > len(p):
            prefix = p
            stem = word[len(p):]
            break

    # Detect Suffix
    for s in sorted(suffixes, key=len, reverse=True):
        if stem.endswith(s) and len(stem) > len(s):
            suffix = s
            stem = stem[:-len(s)]
            break

    porter_stem = porter.stem(word)
    snowball_stem = snowball.stem(word)

    # Lemmatization
    lemma = lemmatizer.lemmatize(word)

    return [
        word,
        prefix,
        stem,
        suffix,
        porter_stem,
        snowball_stem,
        lemma
    ]

# ----------------------------------------------------------
# Display Results
# ----------------------------------------------------------
print("=" * 110)
print("{:<18}{:<12}{:<18}{:<12}{:<18}{:<18}{:<18}".format(
    "Word", "Prefix", "Stem", "Suffix",
    "Porter Stem", "Snowball Stem", "Lemma"))
print("=" * 110)

for w in words:
    result = analyze_word(w)
    print("{:<18}{:<12}{:<18}{:<12}{:<18}{:<18}{:<18}".format(*result))

print("=" * 110)



# ============================================================
# Comparing Word Representation Techniques
# One-Hot Encoding, Bag of Words, TF-IDF, Word2Vec and GloVe
# ============================================================

# Install libraries if required:
# !pip install scikit-learn gensim numpy

print("Rishabh Raikwar 23BAI70136")

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models import Word2Vec
import gensim.downloader as api


# ============================================================
# 2. CORPUS
# ============================================================

corpus = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are great pets"
]

print("Original Corpus:")
for i, sentence in enumerate(corpus, 1):
    print(i, ":", sentence)


# ============================================================
# 3. VOCABULARY
# ============================================================

vocab = [
    'and', 'are', 'cat', 'cats', 'dog', 'dogs',
    'great', 'log', 'mat', 'on', 'pets', 'sat', 'the'
]

print("\nVocabulary:")
print(vocab)

print("\nVocabulary Size:", len(vocab))


# ============================================================
# 4. ONE-HOT ENCODING
# ============================================================

print("\n" + "=" * 60)
print("4. ONE-HOT ENCODING")
print("=" * 60)

words_array = np.array(vocab).reshape(-1, 1)

ohe = OneHotEncoder(sparse_output=False)

onehot_vectors = ohe.fit_transform(words_array)

print("\nOne-Hot Vector for 'cat':")

cat_index = vocab.index("cat")

print(onehot_vectors[cat_index])

print("\nDimensions:")
print(onehot_vectors.shape)


# ============================================================
# 5. BAG OF WORDS
# ============================================================

print("\n" + "=" * 60)
print("5. BAG OF WORDS")
print("=" * 60)

bow_vectorizer = CountVectorizer()

bow_matrix = bow_vectorizer.fit_transform(corpus)

print("\nVocabulary:")
print(bow_vectorizer.get_feature_names_out())

print("\nBag of Words Matrix:")

print(bow_matrix.toarray())


# ============================================================
# 6. TF-IDF
# ============================================================

print("\n" + "=" * 60)
print("6. TF-IDF")
print("=" * 60)

tfidf_vectorizer = TfidfVectorizer()

tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

print("\nVocabulary:")
print(tfidf_vectorizer.get_feature_names_out())

print("\nTF-IDF Matrix:")

print(np.round(tfidf_matrix.toarray(), 3))


# ============================================================
# 7. WORD2VEC
# ============================================================

print("\n" + "=" * 60)
print("7. WORD2VEC")
print("=" * 60)

# Convert sentences into tokenized form
tokenized_corpus = [
    sentence.lower().split()
    for sentence in corpus
]

print("\nTokenized Corpus:")
print(tokenized_corpus)


# Train Word2Vec model
w2v_model = Word2Vec(
    sentences=tokenized_corpus,
    vector_size=10,
    window=3,
    min_count=1,
    sg=1,
    epochs=200,
    seed=42
)


# Display vocabulary
print("\nWord2Vec Vocabulary:")
print(w2v_model.wv.index_to_key)


# Get vector for cat
vector = w2v_model.wv["cat"]

print("\nVector for 'cat':")
print(np.round(vector, 3))

print("\nVector Dimension:")
print(len(vector))


# Similar words
similar = w2v_model.wv.most_similar(
    "cat",
    topn=3
)

print("\nWords Most Similar to 'cat':")

for word, score in similar:
    print(word, ":", round(score, 3))


# ============================================================
# 8. GLOVE
# ============================================================

print("\n" + "=" * 60)
print("8. GLOVE")
print("=" * 60)

print("\nLoading Pretrained GloVe Vectors...")

glove_model = api.load("glove-wiki-gigaword-50")

print("GloVe Model Loaded Successfully!")


# Get vector for cat
glove_vector = glove_model["cat"]

print("\nGloVe Vector for 'cat':")

# Display first 10 values
print(np.round(glove_vector[:10], 3))

print("\nGloVe Vector Dimension:")
print(len(glove_vector))


# Similar words
glove_similar = glove_model.most_similar(
    "cat",
    topn=3
)

print("\nWords Most Similar to 'cat':")

for word, score in glove_similar:
    print(word, ":", round(score, 3))


# ============================================================
# 9. COSINE SIMILARITY - WORD2VEC
# ============================================================

print("\n" + "=" * 60)
print("9. WORD2VEC COSINE SIMILARITY")
print("=" * 60)

w2v_similarity = w2v_model.wv.similarity(
    "cat",
    "dog"
)

print("\nWord2Vec Similarity between 'cat' and 'dog':")

print(round(w2v_similarity, 3))


# ============================================================
# 10. COSINE SIMILARITY - GLOVE
# ============================================================

print("\n" + "=" * 60)
print("10. GLOVE COSINE SIMILARITY")
print("=" * 60)

glove_similarity = glove_model.similarity(
    "cat",
    "dog"
)

print("\nGloVe Similarity between 'cat' and 'dog':")

print(round(glove_similarity, 3))


# ============================================================
# 11. FINAL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("11. FINAL COMPARISON")
print("=" * 60)

print("\nTechnique Comparison:")

print("-----------------------------------------------")
print("Technique       Type          Meaning")
print("-----------------------------------------------")
print("One-Hot         Sparse        No")
print("BoW             Sparse        No")
print("TF-IDF          Sparse        No")
print("Word2Vec        Dense         Yes")
print("GloVe           Dense         Yes")
print("-----------------------------------------------")


# ============================================================
# 12. WORD2VEC VS GLOVE
# ============================================================

print("\n" + "=" * 60)
print("12. WORD2VEC VS GLOVE")
print("=" * 60)

print("\nWord2Vec:")
print("Learns word representations from local context.")

print("\nGloVe:")
print("Learns word representations from global")
print("word co-occurrence statistics.")

print("\nWord2Vec 'cat' vs 'dog':",
      round(w2v_similarity, 3))

print("GloVe 'cat' vs 'dog':",
      round(glove_similarity, 3))


# ============================================================
# END
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETED")
print("=" * 60)




# ============================================
# BIGRAM LANGUAGE MODEL
# Zero-Frequency Problem & Add-1 Smoothing
# ============================================

from collections import Counter
print("Adarsh, 23bai70091")

# Step 1: Training data
text = "I love NLP I love Python I like AI"

# Convert to lowercase and split into words
words = text.lower().split()

print("Words:")
print(words)


# Step 2: Create bigrams
bigrams = []

for i in range(len(words) - 1):
    bigrams.append((words[i], words[i + 1]))

print("\nBigrams:")
print(bigrams)


# Step 3: Count bigrams
bigram_counts = Counter(bigrams)

print("\nBigram Counts:")
print(bigram_counts)


# Step 4: Count unigrams
unigram_counts = Counter(words)

print("\nUnigram Counts:")
print(unigram_counts)


# Step 5: Vocabulary size
vocabulary = set(words)
V = len(vocabulary)

print("\nVocabulary:")
print(vocabulary)

print("Vocabulary Size:", V)


# Step 6: Bigram probability without smoothing
def bigram_probability(w1, w2):
    return bigram_counts[(w1, w2)] / unigram_counts[w1]


print("\nWithout Smoothing:")

print("P(love | i) =",
      bigram_probability("i", "love"))

print("P(ai | love) =",
      bigram_probability("love", "ai"))


# Step 7: Add-1 (Laplace) smoothing
def smoothed_bigram_probability(w1, w2):
    return (bigram_counts[(w1, w2)] + 1) / \
           (unigram_counts[w1] + V)


print("\nWith Add-1 Smoothing:")

print("P(love | i) =",
      smoothed_bigram_probability("i", "love"))

print("P(ai | love) =",
      smoothed_bigram_probability("love", "ai"))


# Step 8: Comparison
print("\nComparison:")

print("Without smoothing:",
      bigram_probability("love", "ai"))

print("With Add-1 smoothing:",
      smoothed_bigram_probability("love", "ai"))

# ============================================
# MLE vs Laplace vs Kneser-Ney
# ============================================

from nltk.lm import MLE, Laplace, KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline


# Training corpus
corpus = [
    "the cat sat on the mat".split(),
    "the dog sat on the rug".split(),
    "the cat chased the mouse".split(),
    "the dog chased the ball".split()
]

# Trigram model
N = 3


# Function to train model
def train(model):

    train_data, vocab = padded_everygram_pipeline(
        N, corpus
    )

    model.fit(train_data, vocab)

    return model


# Train three models
models = {
    "MLE": train(MLE(N)),
    "Laplace": train(Laplace(N)),
    "KneserNey": train(KneserNeyInterpolated(N))
}


# Test cases
tests = [
    ("cat", ["the"]),
    ("mouse", ["the", "chased"]),
    ("rug", ["the", "cat"])
]


# Display probabilities
for name, model in models.items():

    print("\n", name)

    for word, context in tests:

        probability = model.score(
            word,
            context
        )

        print(
            word,
            "|",
            context,
            "=",
            round(probability, 6)
        )