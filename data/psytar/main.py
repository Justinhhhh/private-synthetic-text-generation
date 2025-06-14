import json
import random

if __name__ == "__main__":

    file_original = "./original.jsonl"
    output_dir = "./psytar"

    label_map = {
        "ADR": "Adverse Drug Reaction",
        "DI": "Drug Indications",
        "EF": "Drug Effectiveness",
        "INF": "Drug Ineffeciveness",
        "Others": "Others",
        "SSI": "Sign/Symptoms/Illness",
        "WD": "Withdrowal Symptoms"
    }
    label_number = {
        "ADR": 0,
        "DI": 1,
        "EF": 2,
        "INF": 3,
        "Others": 6,
        "SSI": 4,
        "WD": 5
    }
    converted_data = []

    with open(file_original, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            label_names = [label_map[l] for l in data['label']]
            prompt = f"Write a {', '.join(label_names)} review:"
            converted_data.append({
                "src": prompt,
                "trg": data["text"]
            })


    total = len(converted_data)
    val_size = int(0.1 * total)
    test_size = int(0.1 * total)
    train_size = total - val_size - test_size

    val_data = converted_data[:val_size]
    test_data = converted_data[val_size:val_size + test_size]
    train_data = converted_data[val_size + test_size:]

    test_data = [{"src": item["src"], "trg": ""} for item in test_data]


    def write_jsonl(data, path):
        with open(path, "w") as f:
            for item in data:
                json.dump(item, f)
                f.write('\n')
    sample_data = [{"src": item["src"], "trg": ""} for item in converted_data]
    write_jsonl(train_data, f"{output_dir}/train.jsonl")
    write_jsonl(val_data, f"{output_dir}/validation.jsonl")
    write_jsonl(test_data, f"{output_dir}/test.jsonl")
    write_jsonl(sample_data, f"{output_dir}/samples.jsonl")
    print(f"Train: {len(train_data)}")
    print(f"Val:   {len(val_data)}")
    print(f"Test:  {len(test_data)} (trg='')")
