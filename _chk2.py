import json
a = json.load(open("results/archive/ablation_5run.json"))
print("keys:", list(a.keys()))
for r in a["results"]:
    if "corruption" in r["question"]:
        c = r["comparison"]
        print(r["question"], "| mean_a=", c["mean_a"], "mean_b=", c["mean_b"])
        print("   a_desc mean:", r["a_desc"].get("mean"), " b_desc mean:", r["b_desc"].get("mean"))
