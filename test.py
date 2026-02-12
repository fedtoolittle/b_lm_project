from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")

text = "User: According to the passage, who quarrelled at the start of the poem?"
encoded = tokenizer.encode(text)

print(encoded.tokens)
print(encoded.ids)

decoded = tokenizer.decode(encoded.ids)
print(decoded)
