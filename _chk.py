import json
d = json.load(open("results/ablation.json"))
for r in d["results_main"]:
    c = r["comparison"]
    print("%-42s ci_lo=%-22r ci_hi=%-22r" % (r["question"], c["ci_lo"], c["ci_hi"]))
print("archive?", __import__("os").path.exists("results/archive/ablation_5run.json"))
a = json.load(open("results/archive/ablation_5run.json"))
print("31.258 in archive:", "31.258" in json.dumps(a))
