import csv


def read_csv_with_header(path: str) -> list:
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        result = [item for item in reader]
    return result
