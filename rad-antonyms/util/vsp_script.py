import io
import random
from datetime import datetime

import numpy as np

with io.open('ro_ft_300.vec', encoding="utf-8") as vectors_file:
    with io.open('vocab.txt', encoding="utf-8") as vocab_file:

        vocab = list()
        for line in vocab_file:
            vocab.append(line.strip())

        vectors = dict()
        dimensions = str(next(vectors_file))
        if len(dimensions.split(" ")) == 2:
            pass
        else:
            line = dimensions.split(' ', 1)
            key = line[0]
            if key in vocab:
                vectors[key] = np.fromstring(line[1], dtype="float32", sep=' ')
            dimensions = None
        for line in vectors_file:
            line = line.split(' ', 1)
            key = line[0]
            if key in vocab:
                vectors[key] = np.fromstring(line[1], dtype="float32", sep=' ')
        rho = 0.2

        print(f"Computing VSP pairs @ {datetime.now()}")
        vsp_pairs = dict()
        th = 1 - rho
        vocab = list(vocab)
        words_count = len(vocab)

        step_size = 1000
        vec_size = random.choice(list(vectors.values())).shape[0]

        # List of ranges of step size
        ranges = list()

        left_range_limit = 0
        while left_range_limit < words_count:
            # Create tuple of left range -> right range (min between nr of words (maximum) or left limit + step)
            current_range = (left_range_limit, min(words_count, left_range_limit + step_size))
            ranges.append(current_range)
            left_range_limit += step_size

        range_count = len(ranges)
        print("Started computing VSP pairs")

        for left_range in range(range_count):
            print(f"Left range: {left_range} / {range_count}")
            for right_range in range(left_range, range_count):
                print(f"\tRight range: {right_range} / {range_count}")
                left_translation = ranges[left_range][0]
                right_translation = ranges[right_range][0]

                vecs_left = np.zeros((step_size, vec_size), dtype="float32")
                vecs_right = np.zeros((step_size, vec_size), dtype="float32")

                full_left_range = range(ranges[left_range][0], ranges[left_range][1])
                full_right_range = range(ranges[right_range][0], ranges[right_range][1])

                for index in full_left_range:
                    vecs_left[index - left_translation, :] = vectors[vocab[index]]

                for index in full_right_range:
                    vecs_right[index - right_translation, :] = vectors[vocab[index]]

                dot_product = vecs_left.dot(vecs_right.T)
                indices = np.where(dot_product >= th)

                pairs_count = indices[0].shape[0]
                left_indices = indices[0]
                right_indices = indices[1]

                for index in range(0, pairs_count):
                    left_word = vocab[left_translation + left_indices[index]]
                    right_word = vocab[right_translation + right_indices[index]]

                    if left_word != right_word:
                        score = 1 - dot_product[left_indices[index], right_indices[index]]
                        vsp_pairs[(left_word, right_word)] = score
                        vsp_pairs[(right_word, left_word)] = score

        print("Finished computing VSP pairs. Writing")
        with io.open(file="vsp_pairs.txt", mode="w", encoding="utf-8") as output_file:
            for k in vsp_pairs.keys():
                output_file.write(f"{k[0]} {k[1]}")
            output_file.close()
        vocab_file.close()
    vectors_file.close()
