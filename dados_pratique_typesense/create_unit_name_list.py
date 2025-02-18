import json

# Read the unidades.json file
with open('unidades.json', 'r', encoding='utf-8') as f:
    unidades_data = json.load(f)

# Extract just the unit_ids into a list
unit_ids = [unit['id'] for unit in unidades_data]

# Write the list to a new JSON file
with open('unit_ids.json', 'w', encoding='utf-8') as f:
    json.dump(unit_ids, f, ensure_ascii=False, indent=2)
