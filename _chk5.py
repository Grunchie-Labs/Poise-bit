import json
d = json.load(open("results/tuned.json"))
for p in d["tuning"]["Rastrigin/dim=2/clean"]["pbit"]["grid"]:
    if p["params"]["lr"] in (0.05, 0.005):
        print(p["params"], round(p["score"], 6))
