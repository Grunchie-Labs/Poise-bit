import json
for tag, f in ((chr(48)+chr(48), "results/ablation.json"), ("1", "results/ablation_seed1_v2.json")):
    d = json.load(open(f))
    print("=== seed", tag, " separation_valid=", d["metadata"]["separation_valid"])
    print("  derived:", {k: round(v, 6) for k, v in d["derived"].items()})
    for r in d["validity_report"]:
        print("  VALID %-22s rel=%.3g bound=%.3g %.2fx %s" % (
            r["cell"], r["rel_diff"], r["bound"], r["ratio_to_bound"],
            "OK" if r["within_bound"] else "EXCEEDS"))
