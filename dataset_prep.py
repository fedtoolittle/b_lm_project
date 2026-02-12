# import json
# import random

# INPUT_FILE = "dataset.jsonl"
# OUTPUT_FILE = "train_text.txt"

# with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
#      open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
    
#     for line in f_in:
#         obj = json.loads(line)
        
#         question = obj["qText"].strip()
#         answers = obj["answers"]
        
#         if not answers:
#             continue
        
#         answer = random.choice(answers).strip()
        
#         formatted = (
#             f"User: {question}\n"
#             f"Assistant: {answer}\n\n"
#         )
        
#         f_out.write(formatted)

# print("Dataset prepared.")

import json
import re

output_lines = []

with open("dataset.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)

        q = obj["qText"].strip()
        q = re.sub(r"\s*\((detail|reference|case|instance|example)\s+\d+\)", "", q)
        a = obj["answers"][0].strip()

        formatted = (
            f"<|question|> {q}\n"
            f"<|answer|> {a} <|eos|>\n"
        )

        output_lines.append(formatted)

with open("data.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))
print ("Dataset prepared.")