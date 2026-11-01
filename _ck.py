import json
d=json.load(open('results/claims.json'))
c={x['cid']:x for x in d['results']}
for k in ('C1','C2','C3a','C3b','C4','C7'):
    print('%-4s raw=%-9s holm=%-9s %s' % (k, round(c[k]['comparison']['p_value'],5), round(c[k]['comparison']['p_value_holm'],5), c[k]['supports_hypothesis']))
