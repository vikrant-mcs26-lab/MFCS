import re
import numpy as np

def extract_alphanumerical(paragraph: str) -> list[str]:
    regex = "[^a-zA-Z0-9 ]" # Only allow alphanumerical values and whitespace.
    return re.sub(regex, ' ', paragraph.lower()).split()

def get_frequency_vector(paragraph: list[str], known_words: list[str]) -> np.ndarray:
    out_arr = np.zeros((len(known_words), 1))

    for word in paragraph:
        idx = known_words.index(word)
        out_arr[idx] += 1

    return out_arr

def get_unique_words(paragraph: list[str]) -> list[str]:
    return list(set(paragraph))

def compute_similarity(matrix: np.ndarray, para1_idx: int, para2_idx: int) -> dict:
    vec1 = matrix[:, para1_idx]
    vec2 = matrix[:, para2_idx]

    dot_similarity = np.dot(vec1, vec2)
    cosine_similarity = dot_similarity / (np.sqrt(np.dot(vec1, vec1)) * np.sqrt(np.dot(vec2, vec2)))

    cont = vec1 * vec2
    contribution = np.argsort(cont)[-5:]

    return {"dot_similarity":dot_similarity, "cosine_similarity":cosine_similarity, "contributions":contribution}