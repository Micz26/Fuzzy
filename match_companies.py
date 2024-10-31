from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

from flashtext import KeywordProcessor
import string


try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


list_a = ["Apple inc", "Msft", "Meta"]

list_b = [
    "Apple Computer Incoporated",
    "NVIDIA Corporation",
    "Microsoft",
    "Apple compter Inc.",
    "mcsft",
    "tesla",
    "Microsoft Corporation",
    "Microsof corp.",
    "Meta Platforms, Inc.",
    "meta incorporated",
    "appl",
    "Google LLC",
    "Computer Inc Apple",
    "Apple LLC",
    "Microsot LLC",
]

keyword_processor = KeywordProcessor()

keyword_processor.add_keyword("inc", "incorporated")
keyword_processor.add_keyword("llc", "limited liability company")
keyword_processor.add_keyword("msft", "microsoft")
keyword_processor.add_keyword("appl", "apple")


def process_company_name(company):
    company = company.translate(str.maketrans("", "", string.punctuation)).lower()
    company = keyword_processor.replace_keywords(company)

    words = company.split()

    processed_words = [stemmer.stem(word) for word in words if word not in stop_words]

    return " ".join(processed_words)


list_a_dict = {word: process_company_name(word) for word in list_a}
list_b_dict = {word: process_company_name(word) for word in list_b}

vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
list_b_vectors = vectorizer.fit_transform(list_b_dict.values())
list_a_vectors = vectorizer.transform(list_a_dict.values())

nbrs = NearestNeighbors(n_neighbors=3, metric="cosine").fit(list_b_vectors)


def match_companies():
    distances, indices = nbrs.kneighbors(list_a_vectors)

    list_b_keys = list(list_b_dict.keys())

    results = {}
    for i, company_a in enumerate(list_a_dict.keys()):
        results[company_a] = [
            (list_b_keys[idx], 1 - distances[i][j]) for j, idx in enumerate(indices[i])
        ]
        print(f"\nBest matches for '{company_a}':")
        for match, score in results[company_a]:
            print(f"  {match} (Score: {score})")


if __name__ == "__main__":
    match_companies()
