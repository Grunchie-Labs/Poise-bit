import json
d = json.load(open('results/ablation.json'))
print('meta: low_beta=%s high_beta=%s residual=%s valid=%s' % (
    d['metadata']['low_beta'], d['metadata']['high_beta'],
    d['metadata']['residual_flip_prob_at_high_beta'], d['metadata']['separation_valid']))
print()
print('tuning selections (10 tune runs, main-study budget):')
for k, v in d['tuning'].items():
    print('  %-17s %-34s score=%.4f' % (k, v['selected'], v['selected_score']))
print()
for r in d['results_main']:
    c = r['comparison']
    print('%-44s diff=%+8.4f  CI=[%+.3f,%+.3f] raw=%.4g holm=%.4g' % (
        r['question'], c['mean_diff'], c['ci_lo'], c['ci_hi'], c['p_value'], c['p_value_holm']))
print()
print('per-arm clean means:')
for r in d['results_main']:
    if r['cell'] != 'clean':
        continue
    print('  %-17s %9.4f' % (r['a'], r['a_desc']['mean']))
print('  %-17s %9.4f' % (d['results_main'][0]['b'], d['results_main'][0]['b_desc']['mean']))
