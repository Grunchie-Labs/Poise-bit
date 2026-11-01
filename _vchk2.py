import json
d = json.load(open("results/claims.json"))
for r in d["results"]:
    if r["cid"] in ("C2", "C1", "C7"):
        print(r["cid"], json.dumps(r["comparison"], indent=1))
