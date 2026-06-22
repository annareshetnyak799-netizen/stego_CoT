#!/usr/bin/env python3
"""Regenerate a clearer figure.png for a read-back run, offline (no API).
Usage: python replot.py <run_dir>
The story in 3 panels:
  A) where the next letter lands, per condition: on the read-back letter vs the original letter
  B) distribution shift from untouched (KL), forced vs placebo, with 95% CI
  C) force-vs-placebo gap vs chance (permutation null)"""
import json, sys, glob, pathlib, random, collections, math
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

run = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(sorted(glob.glob('readback_dist/*'))[-1])
MAP = list('SCPABTDMRF'); INV = {L:i for i,L in enumerate(MAP)}
recs = [json.loads(l) for l in open(run/'dists.jsonl')]
st = json.loads((run/'stats.json').read_text())
GO, OR, GREY, RED = '#1f7a4d', '#b3471f', '#cfcabd', '#c0392b'

def frac(dist, letter):
    tot = sum(dist.values()) or 1; return dist.get(letter, 0)/tot
def ognext(r): return MAP[(INV[r['orig_letter']]+1) % 10]
def mean_ci(xs):
    n=len(xs); m=sum(xs)/n
    if n<2: return m,0,0
    sd=(sum((x-m)**2 for x in xs)/(n-1))**0.5; h=1.96*sd/n**0.5
    return m, max(0,m-h), min(1,m+h)

# ---- panel A data: per-trace fraction on read-back letter and on original letter, per condition
conds = ['base','placebo','force']
on_rb = {c: [] for c in conds}; on_og = {c: [] for c in conds}
for r in recs:
    rb, og = r['rb_pred'], ognext(r)
    for c in conds:
        on_rb[c].append(frac(r[c], rb)); on_og[c].append(frac(r[c], og))

fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.3))

# Panel A
labels = ['untouched','placebo','forced']; x = range(3); w = 0.38
rb_m = [mean_ci(on_rb[c]) for c in conds]; og_m = [mean_ci(on_og[c]) for c in conds]
ax[0].bar([i-w/2 for i in x], [m[0] for m in og_m],
          yerr=[[m[0]-m[1] for m in og_m],[m[2]-m[0] for m in og_m]], width=w, capsize=3,
          color=OR, label="followed memorized plan (ignored edit)")
ax[0].bar([i+w/2 for i in x], [m[0] for m in rb_m],
          yerr=[[m[0]-m[1] for m in rb_m],[m[2]-m[0] for m in rb_m]], width=w, capsize=3,
          color=GO, label="adjusted to new sentence (read-back)")
ax[0].set_xticks(list(x)); ax[0].set_xticklabels(labels); ax[0].set_ylim(0,1)
ax[0].set_ylabel('share of sampled continuations')
ax[0].set_title("Memorized plan vs adjusting to the new sentence\n(forced: does it react to the wrong letter?)")
ax[0].legend(fontsize=8, loc='upper center')

# Panel B: KL with CI
mf,lf,hf = st['mean_kl_force']; mp,lp,hp = st['mean_kl_placebo']
ax[1].bar(['forced','placebo'], [mf,mp], yerr=[[mf-lf,mp-lp],[hf-mf,hp-mp]], capsize=5, color=[GO,OR])
ax[1].set_ylabel('KL distance from "untouched"')
ax[1].set_title("How far the WHOLE distribution moved\n(0 = identical to untouched; mean ±95% CI)")
for i,(v,lo) in enumerate([(mf,lf),(mp,lp)]): ax[1].text(i, v+0.06, f'{v:.2f}', ha='center', fontsize=9)

# Panel C: permutation null vs observed
rows = st['rows']; deltas = [x['delta'] for x in rows]; obs = sum(deltas)/len(deltas)
rng = random.Random(1); null=[]
for _ in range(4000):
    d=0.0
    for x in rows:
        a,b=x['kl_force'],x['kl_placebo']
        if rng.random()<0.5: a,b=b,a
        d += a-b
    null.append(d/len(rows))
ax[2].hist(null, bins=40, color=GREY)
ax[2].axvline(obs, color=RED, lw=2)
ax[2].set_xlabel('force−placebo gap'); ax[2].set_yticks([])
ax[2].set_title(f"Is the gap real?\nobserved (red) vs chance shuffles · p={st['perm_p']:.4f}")
ax[2].annotate('observed', (obs, ax[2].get_ylim()[1]*0.9), color=RED, ha='right', fontsize=9)

plt.tight_layout(); plt.savefig(run/'figure.png', dpi=130)
print('wrote', run/'figure.png')
print(f"A: forced on read-back={mean_ci(on_rb['force'])[0]:.0%} vs original={mean_ci(on_og['force'])[0]:.0%}; "
      f"base on original={mean_ci(on_og['base'])[0]:.0%}; placebo on original={mean_ci(on_og['placebo'])[0]:.0%}")
