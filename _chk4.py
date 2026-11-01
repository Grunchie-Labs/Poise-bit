import json
d = json.load(open("results/tuned.json"))
g = d["tuning"]["Rastrigin/dim=2/clean"]
print(json.dumps(g, indent=1)[:900])
