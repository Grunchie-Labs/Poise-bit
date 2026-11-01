import json
for tag, f in (('seed 0','results/ablation.json'), ('seed 1','results/ablation_seed1_v2.json')):
    d = json.load(open(f))
    print('=== %s  separation_valid=%s ===' % (tag, d['metadata']['separation_valid']))
    print('  amplification:', {k: round(v,3) for k,v in d['metadata']['validity_amplification'].items()})
    for r in d['validity_report']:
        print('  %-24s rel=%.3g bound=%.3g  %.2fx  %s' % (r['cell'], r['rel_diff'], r['bound'], r['ratio_to_bound'], 'OK' if r['within_bound'] else 'EXCEEDS'))
