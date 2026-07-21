# test_tokenizer.py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E4B")
print("Tokenizer vocabulary size:", len(tokenizer))

print("Tokenizing choices:")
for choice in ["A", "B", "C", "D"]:
    ids = tokenizer.encode(choice, add_special_tokens=False)
    print(f"'{choice}': {ids} {[tokenizer.decode([i]) for i in ids]}")

for choice in [" A", " B", " C", " D"]:
    ids = tokenizer.encode(choice, add_special_tokens=False)
    print(f"'{choice}': {ids} {[tokenizer.decode([i]) for i in ids]}")

print("Tokenizing templates:")
print("'Answer: A':", tokenizer.encode("Answer: A", add_special_tokens=False))
print("'Answer:A':", tokenizer.encode("Answer:A", add_special_tokens=False))
print("'Answer:  A':", tokenizer.encode("Answer:  A", add_special_tokens=False))
