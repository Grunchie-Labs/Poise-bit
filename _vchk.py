import json
d = json.load(open("results/claims.json"))
for r in d["results"]:
    c = r.get("comparison") or {}
    raw = c.get("p_value", None)
    holm = c.get("p_value_holm", None)
    print("%-5s raw=%-10s holm=%-10s %s" % (r["cid"], ("%.5g" % raw) if raw is not None else "-", ("%.5g" % holm) if holm is not None else "-", r.get("supports_hypothesis", "")))
