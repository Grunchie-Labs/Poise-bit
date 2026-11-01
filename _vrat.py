import json
for tag, f in (("seed 0", "results/ablation.json"), ("seed 1", "results/ablation_seed1_v2.json")):
    d = json.load(open(f))
    print("===", tag, "separation_valid=", d["metadata"]["separation_valid"])
    for v in d["validity_report"]:
        print("  %-22s rel=%-11.4g bound=%-11.4g ratio=%-8.3f within=%s" % (v["cell"], v["rel_diff"], v["bound"], v["ratio_to_bound"], v["within_bound"]))
