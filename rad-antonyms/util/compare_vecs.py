import io

import numpy as np
from numpy import dot
from numpy.linalg import norm

from util import tools

ORIGINAL_VECS = 'lang/vectors/ro_ft_300.vec'

CF_VECS = 'lang/vectors/ro_ft_300_allpos_datasets_ant_syn_vsp_aug2.vec'
VOCABULARY = 'lang/vocab_small_diac.txt'

dataset_antonyms = [
    ("mărești", "micșorează"),
    ("mărești", "diminuează"),
    ("porni", "opri"),
    ("porni", "stinge"),
    ("inchide", "deschide")
]


def cosine_sim(a, b):
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim


def compare_vecs(original_vecs_path, cf_vecs_path, vocab_path):
    vocab = set()
    with open(file=vocab_path, mode="r", encoding="utf-8") as vocab_file:
        for line in vocab_file:
            vocab.add(line.strip())

    # Load the word vectors
    original_dim, original_vecs = tools.load_vectors(original_vecs_path, vocab)

    # Load the new vectors
    new_dim, new_vecs = tools.load_vectors(cf_vecs_path, vocab)

    identical = True
    for k, v in new_vecs.items():
        try:
            old_values = original_vecs[k]
        except KeyError:
            print(f"{k} not foudn in original vecs")
            identical = False
            break
        if not np.array_equal(v, old_values):
            cos = cosine_sim(v, old_values)
            print('{} diff value -- cosine sim {}'.format(k, cos))
            identical = False
    print("Identical vector values" if identical else "Not identical vector values")


def report_vectors(og_path, cf_path):
    _, og_vecs = tools.load_vectors_novocab(og_path)
    _, cf_vecs = tools.load_vectors_novocab(cf_path)
    while True:
        w1, w2 = str(input()).split(" ")
        print(f"Cos sim for {w1} between original and counterfit embedding {cosine_sim(og_vecs[w1], cf_vecs[w1])}"
              f"Cos sim for {w2} between original and counterfit embedding {cosine_sim(og_vecs[w2], cf_vecs[w2])}"
              f"Cos sim between {w1} and {w2} original: {cosine_sim(og_vecs[w1], og_vecs[w2])}"
              f"Cos sim between {w1} and {w2} counterfit: {cosine_sim(cf_vecs[w1], cf_vecs[w2])}")


def compare_antonyms(path1, path2):
    set1 = set()
    set2 = set()
    with io.open(path1, "r") as p1_file:
        for line in p1_file.readlines():
            w1, w2 = line.split(" ")
            set1.add((w1, w2))
    with io.open(path2, "r") as p2_file:
        for line in p2_file.readlines():
            w1, w2 = line.split(" ")
            set2.add((w1, w2))

    for i in set1:
        if i not in set2:
            print(f"{i} in set1 but not 2")

    for i in set2:
        if i not in set1:
            print(f"{i} in set2 but not 1")


def main():
    compare_vecs(ORIGINAL_VECS, CF_VECS, VOCABULARY)


if __name__ == "__main__":
    report_vectors(ORIGINAL_VECS, CF_VECS)
