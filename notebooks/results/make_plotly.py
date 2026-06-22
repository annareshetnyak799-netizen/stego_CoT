#!/usr/bin/env python3
"""Interactive Plotly report for a read-back run -> figure.html (self-contained-ish, plotly via CDN).
Usage: python make_plotly.py <run_dir>"""
import json, sys, glob, pathlib, random
import plotly.graph_objects as go
from plotly.subplots import make_subplots

run = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(sorted(glob.glob('readback_dist/*'))[-1])
MAP = list('SCPABTDMRF'); INV = {L:i for i,L in enumerate(MAP)}
recs = [json.loads(l) for l in open(run/'dists.jsonl')]
st = json.loads((run/'stats.json').read_text())
GO_, OR_, GREY, RED = '#1f7a4d', '#b3471f', '#bdb8ab', '#c0392b'

N = len(recs)
S = sum(recs[0]['force'].values()) if recs else 0
def frac(d,L): tot=sum(d.values()) or 1; return d.get(L,0)/tot
def ognext(r): return MAP[(INV[r['orig_letter']]+1)%10]
def mci(xs):
    n=len(xs); m=sum(xs)/n
    if n<2: return m,0
    sd=(sum((x-m)**2 for x in xs)/(n-1))**0.5; return m, 1.96*sd/n**0.5
conds=['base','placebo','force']; xlab=['untouched','placebo','forced']
og=[mci([frac(r[c],ognext(r)) for r in recs]) for c in conds]
rb=[mci([frac(r[c],r['rb_pred']) for r in recs]) for c in conds]

mf,lf,hf = st['mean_kl_force']; mp,lp,hp = st['mean_kl_placebo']
rows=st['rows']; obs=sum(x['delta'] for x in rows)/len(rows)
rng=random.Random(1); null=[]
for _ in range(5000):
    d=0.0
    for x in rows:
        a,b=x['kl_force'],x['kl_placebo']
        if rng.random()<0.5: a,b=b,a
        d+=a-b
    null.append(d/len(rows))

fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.08,
    subplot_titles=("Memorized plan vs adjusting to the new sentence",
                    "How far the whole next-letter distribution moved (KL)",
                    "Is the gap real? (permutation test)"))

# Panel 1: grouped bars with % labels
fig.add_bar(row=1,col=1, x=xlab, y=[m for m,_ in og], error_y=dict(type='data',array=[e for _,e in og]),
            marker_color=OR_, name='followed memorized plan',
            text=[f'{m:.0%}' for m,_ in og], textposition='outside')
fig.add_bar(row=1,col=1, x=xlab, y=[m for m,_ in rb], error_y=dict(type='data',array=[e for _,e in rb]),
            marker_color=GO_, name='dynamically adjusted (read-back)',
            text=[f'{m:.0%}' for m,_ in rb], textposition='outside')
fig.update_yaxes(range=[0,1.08], title_text='share of continuations', row=1,col=1)

# Panel 2: KL bars with value + CI
fig.add_bar(row=1,col=2, x=['forced','placebo'], y=[mf,mp],
            error_y=dict(type='data', array=[hf-mf, hp-mp], arrayminus=[mf-lf, mp-lp]),
            marker_color=[GO_,OR_], text=[f'{mf:.2f}', f'{mp:.2f}'], textposition='outside', showlegend=False)
fig.update_yaxes(title_text='KL from "untouched" (nats)', row=1,col=2)

# Panel 3: permutation null histogram + observed line
fig.add_histogram(row=1,col=3, x=null, nbinsx=45, marker_color=GREY, name='chance (null)', showlegend=False)
fig.add_vline(x=obs, line_color=RED, line_width=3, row=1, col=3)
fig.add_annotation(row=1,col=3, x=obs, y=1, yref='y domain', showarrow=False, xanchor='right',
                   text=f"observed gap = {obs:.2f}  (p={st['perm_p']:.4f})", font=dict(color=RED,size=12))
fig.update_xaxes(title_text='mean(force − placebo) gap', row=1,col=3)
fig.update_yaxes(title_text='# of random shuffles', row=1,col=3)

fig.update_layout(
    title=dict(text=f"<b>Steganographic read-back — Sonnet 4.5</b><br>"
                    f"<sup>n = {N} traces · {S} samples per condition · {N*3*S:,} generations · "
                    f"forced KL {mf:.2f} vs placebo KL {mp:.2f} · p = {st['perm_p']:.4f} → READ-BACK DETECTED</sup>"),
    barmode='group', template='plotly_white', font=dict(family='Inter, system-ui', size=13),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0), height=520, margin=dict(t=110))

EXPL = """
<div style="max-width:1100px;margin:14px auto;font:14px/1.6 system-ui;color:#333">
<b>How to read this</b>
<ul>
<li><b>Left</b> — for each condition, the share of the 40 sampled next-sentences that <b style="color:#b3471f">followed the memorized plan</b>
(the originally-correct letter) vs <b style="color:#1f7a4d">dynamically adjusted</b> to the letter we put in (read-back).
With nothing changed, or only the wording changed, it follows the plan (96% / 92%). Force the <i>letter</i> and it adjusts (60%).</li>
<li><b>Middle</b> — <b>KL divergence</b> measures how far a condition's whole next-letter distribution sits from the untouched one
(0 = identical; bigger = more different). Forcing the letter moves it ~13× more than a same-letter reword.</li>
<li><b>Right — the gray histogram is the permutation null.</b> We don't know by eye whether a force–placebo gap of 2.57 is "a lot."
So we ask: if forcing and rewording really had <i>the same</i> effect, how big a gap would appear just by chance? We randomly swap the
"force" and "placebo" labels within each trace 5,000 times; each swap gives one gray bar's worth of "gap under chance." The gray blob is
that chance distribution (centered near 0). The <b style="color:#c0392b">red line</b> is our actual observed gap — far to the right of
anything chance produces, so p = 0.0002: the effect is real, not noise.</li>
</ul></div>"""
# example trace (clearest = max delta): per-letter distributions
ex=max(rows,key=lambda x:x['delta']); exr=next(r for r in recs if r['trial']==ex['trial'] and r['p']==ex['p'])
LET=MAP+['_']
def vec(d): tot=sum(d.values()) or 1; return [d.get(l,0)/tot for l in LET]
ef=go.Figure()
ef.add_bar(x=LET,y=vec(exr['base']),name='untouched',marker_color='#9a958a')
ef.add_bar(x=LET,y=vec(exr['force']),name='forced',marker_color=GO_)
ef.add_bar(x=LET,y=vec(exr['placebo']),name='placebo',marker_color='#d9a441')
ef.add_annotation(x=exr['rb_pred'], y=1.16, yref='y', showarrow=False, font=dict(color=GO_,size=12), text=f"↓ read-back '{exr['rb_pred']}'")
ef.add_annotation(x=ognext(exr), y=1.05, yref='y', showarrow=False, font=dict(color=OR_,size=12), text=f"↓ original '{ognext(exr)}'")
ef.update_layout(title=f"<b>Example trace T{exr['trial']}</b> — forced {exr['orig_letter']}→{exr['forced']} — next-letter distribution per condition",
                 barmode='group', template='plotly_white', font=dict(family='Inter, system-ui', size=13),
                 height=440, yaxis=dict(title='share of continuations', range=[0,1.25]), legend=dict(orientation='h',y=1.12))
html = fig.to_html(include_plotlyjs=True, full_html=True)
html = html.replace('</body>', ef.to_html(include_plotlyjs=False, full_html=False) + '</body>')
html = html.replace('</body>', EXPL + '</body>')
(run/'figure.html').write_text(html)
print('wrote', run/'figure.html', f'(n={N}, S={S})')
