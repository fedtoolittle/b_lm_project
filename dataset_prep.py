import json
import random

INPUT_FILE = "dataset.jsonl"
OUTPUT_FILE = "train_text.txt"

with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
    
    for line in f_in:
        obj = json.loads(line)
        
        question = obj["qText"].strip()
        answers = obj["answers"]
        
        if not answers:
            continue
        
        answer = random.choice(answers).strip()
        
        formatted = (
            f"User: {question}\n"
            f"Assistant: {answer}\n\n"
        )
        
        f_out.write(formatted)

print("Dataset prepared.")
