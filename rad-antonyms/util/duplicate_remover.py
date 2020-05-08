import io


def remove_duplicates(path: str) -> None:
    unique_pairs = set()
    with io.open(file=path, mode="r", encoding="utf-8") as input_file:
        for each in input_file:
            line = each.strip()
            unique_pairs.add(tuple(sorted(tuple(line.split(" ")))))
    unique_pairs = sorted(unique_pairs, key= lambda x: x[0])
    for pair in unique_pairs:
        print(f"{pair[0]} {pair[1]}")


if __name__ == '__main__':
    remove_duplicates("../lang/dataset_synonyms.txt")
