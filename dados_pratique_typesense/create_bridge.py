import json

# Load the units data from unidades.json
with open('unidades.json', 'r', encoding='utf-8') as f:
    unidades = json.load(f)

# Load the plans data from planos.json
with open('planos.json', 'r', encoding='utf-8') as f:
    planos = json.load(f)

bridge_documents = []

# Create a bridge document for each unit-plan combination.
# Adjust pricing override logic here if needed.
for unit in unidades:
    for plan in planos:
        doc = {
            "id": f"{unit['id']}__{plan['id']}",
            "unit_id": unit['id'],
            "unit_name": unit['name'],
            "unit_city": unit.get('city', ''),
            "unit_state": unit.get('state', ''),
            "unit_type": unit.get('type', ''),
            "plan_id": plan['id'],
            "plan_name": plan['name'],
            # Default local_price is taken from the plan's base price.
            "local_price": plan.get('price', 0.0),
            "local_comment": ""  # You can customize this if there are unit-specific pricing notes.
        }
        bridge_documents.append(doc)

# Write the resulting list to a JSON file
with open('unit_plans.json', 'w', encoding='utf-8') as f:
    json.dump(bridge_documents, f, ensure_ascii=False, indent=2)

print(f"Generated {len(bridge_documents)} unit-plan documents in 'unit_plans.json'")
