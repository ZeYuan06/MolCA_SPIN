import json

with open('data/ChEBI-20_data/validation.txt', 'r') as f:
    raw_lines = f.readlines()

with open('all_checkpoints/validation_generation/lightning_logs/version_0/predictions.txt', 'r') as f:
    add_lines = f.readlines()

new_data = []

new_data.append("CID    SMILES     real    generated")

for raw_line, add_line in zip(raw_lines[1:], add_lines):
    raw_line = raw_line.strip()
    add_data = json.loads(add_line.strip())
    gene_content = add_data.get("prediction", "").strip()

    new_line = f"{raw_line}\t{gene_content}"
    new_data.append(new_line)

with open("data/ChEBI-20_data/validation_convert.txt", 'w') as f:
    f.write("\n".join(new_data))

print("数据添加完成")