#!/usr/bin/env python3
"""Turn a read-back-distribution run folder into a self-contained HTML viewer (Inspect-AI style).

One row per PROMPT (trace x position). The detail pane shows the prompt once, then the sampled
generations GROUPED by condition (untouched / forced / placebo) and by outcome
(read-back / original / off-chain), each expandable.

Usage:  python make_viewer.py [run_dir]      # default: latest results/readback_dist/<ts>/
"""
import json, sys, glob, pathlib, collections

HERE = pathlib.Path(__file__).parent
def latest_run():
    runs = sorted(glob.glob(str(HERE/'readback_dist'/'*')))
    return runs[-1] if runs else None
run = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else latest_run() or '')
if not run.exists(): sys.exit('no run folder found under results/readback_dist/')
def load(name):
    p = run/name
    return [json.loads(l) for l in open(p)] if p.exists() else []

dists = load('dists.jsonl'); raws = load('raw_samples.jsonl')
stats = json.loads((run/'stats.json').read_text()) if (run/'stats.json').exists() else {}
btraces = load('base_traces.jsonl')
MAP = list('SCPABTDMRF'); INV = {L:i for i,L in enumerate(MAP)}
expected = {b['trial']: [MAP[d] for d in b['seq']] for b in btraces}
dkey = {(d['trial'], d['p']): d for d in dists}

# group raw samples by (trial, p, cond)
bucket = collections.defaultdict(lambda: collections.defaultdict(list))
for r in raws:
    bucket[(r['trial'], r['p'])][r['cond']].append({'letter': r['letter'], 'cont': r['continuation']})

groups = []
for (trial, p), conds in sorted(bucket.items()):
    d = dkey.get((trial, p), {})
    exp = expected.get(trial, [])
    orig_next = exp[p] if p < len(exp) else MAP[(INV.get(d.get('orig_letter','S'),0)+1)%10]
    rb_pred = d.get('rb_pred','')
    for cond, samps in conds.items():
        for s in samps:
            if cond == 'force':
                s['outcome'] = 'readback' if s['letter']==rb_pred else 'original' if s['letter']==orig_next else 'offchain'
            else:
                s['outcome'] = 'onchain' if s['letter']==orig_next else 'offchain'
    fcount = collections.Counter(s['outcome'] for s in conds.get('force', []))
    groups.append({
        'trial': trial, 'p': p, 'q': d.get('q',''), 'orig': d.get('orig_letter',''),
        'forced': d.get('forced',''), 'rb_pred': rb_pred, 'orig_next': orig_next,
        'expected': exp, 'base_prompt': d.get('base_prompt',''),
        'prefills': d.get('prefills') or {}, 'samples': conds, 'fcount': dict(fcount),
    })

payload = json.dumps({'run': run.name, 'stats': stats, 'groups': groups})

HTML = r"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>read-back log viewer</title>
<style>
:root{--bg:#fbfaf8;--panel:#fff;--ink:#1b1a16;--muted:#6b6760;--rule:#e6e3da;--go:#1f7a4d;--no:#c0392b;--accent:#b3471f;--amber:#a9781f;}
*{box-sizing:border-box;margin:0;padding:0}
body{font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:var(--bg);color:var(--ink);height:100vh;overflow:hidden}
header{display:flex;align-items:center;gap:16px;padding:10px 16px;border-bottom:1px solid var(--rule);background:var(--panel);flex-wrap:wrap}
header .title{font-weight:700} header .run{font-family:ui-monospace,monospace;color:var(--muted);font-size:12px}
.kpi{font-family:ui-monospace,monospace;font-size:12px;color:var(--muted)} .kpi b{color:var(--ink)}
.pill{font-family:ui-monospace,monospace;font-size:12px;padding:2px 9px;border-radius:999px;font-weight:600}
.pill.det{background:#e7f3ec;color:var(--go)} .pill.no{background:#fbeae7;color:var(--no)}
.wrap{display:grid;grid-template-columns:minmax(300px,400px) 1fr;height:calc(100vh - 49px)}
.list{border-right:1px solid var(--rule);overflow:auto;background:var(--panel)}
.row{padding:10px 12px;border-bottom:1px solid #f0eee7;cursor:pointer}
.row:hover{background:#faf8f3} .row.sel{background:#f3efe6}
.row .t{font-family:ui-monospace,monospace;font-size:12px;color:var(--muted)}
.row .q{font-size:13px;margin:2px 0 5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.minibar{display:flex;height:7px;border-radius:4px;overflow:hidden;background:#eee}
.minibar i{display:block} .minibar .rb{background:var(--go)} .minibar .og{background:var(--accent)} .minibar .off{background:#cfcabd}
.row .leg{font-family:ui-monospace,monospace;font-size:10px;color:var(--muted);margin-top:3px}
.detail{overflow:auto;padding:18px 22px}
.dh{font-weight:700;font-size:16px;margin-bottom:4px}
.sub{font-size:13px;color:var(--ink);margin-bottom:10px;line-height:1.5}
.seqbar{display:flex;align-items:center;gap:4px;flex-wrap:wrap;margin:6px 0 16px}
.seqL{font-family:ui-monospace,monospace;font-size:12px;padding:2px 7px;border:1px solid var(--rule);border-radius:6px;background:#fff;color:#555}
.seqL.cur{border-color:var(--accent);color:var(--accent);font-weight:700;background:#fbeae7}
.seqL.nxt{border-color:var(--go);color:var(--go);font-weight:700;background:#e7f3ec}
.arr{color:var(--muted);font-family:ui-monospace,monospace} .seqcap{font-size:11px;color:var(--muted);margin-left:6px}
.msg{border:1px solid var(--rule);border-radius:9px;margin:10px 0;background:var(--panel);overflow:hidden}
.msg .role{font-family:ui-monospace,monospace;font-size:11px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);padding:7px 12px;border-bottom:1px solid var(--rule);background:#faf8f3;display:flex;justify-content:space-between;cursor:pointer}
.msg .body{padding:11px 13px;white-space:pre-wrap;word-break:break-word;font-family:"SF Mono",ui-monospace,Menlo,monospace;font-size:12.5px;line-height:1.6}
.msg.user .role{color:#3a6ea5} .body.hide{display:none}
.cond{margin:16px 0 6px;font-family:ui-monospace,monospace;font-size:13px;font-weight:700}
.cond .badge{font-size:10px;font-weight:600;padding:1px 7px;border-radius:5px;margin-left:6px}
.b-base{background:#eceae3;color:#555}.b-force{background:#e7f3ec;color:var(--go)}.b-placebo{background:#fcf3e0;color:var(--amber)}
.ogrp{border:1px solid var(--rule);border-radius:8px;margin:6px 0;overflow:hidden}
.ohead{display:flex;align-items:center;gap:8px;padding:7px 11px;cursor:pointer;font-family:ui-monospace,monospace;font-size:12px;background:#faf8f3}
.ohead .dot{width:9px;height:9px;border-radius:50%} .dot.readback,.dot.onchain{background:var(--go)} .dot.original{background:var(--accent)} .dot.offchain{background:#b9b4a8}
.ohead .cnt{margin-left:auto;color:var(--muted)}
.gen{border-top:1px solid #f0eee7;padding:9px 12px;font-family:"SF Mono",ui-monospace,Menlo,monospace;font-size:12px;line-height:1.55;white-space:pre-wrap;word-break:break-word}
.firstlet{font-weight:700;border-radius:3px;padding:0 3px}
.fl-readback,.fl-onchain{background:#dff1e6;color:var(--go)} .fl-original{background:#fbe9d8;color:var(--accent)} .fl-offchain{background:#ececec;color:#555}
.empty{color:var(--muted);padding:40px;text-align:center}
.prefill{margin:4px 0 8px;border:1px dashed var(--rule);border-radius:8px;overflow:hidden;background:#fcfbf8}
.plabel{font-family:ui-monospace,monospace;font-size:10px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);padding:6px 11px;border-bottom:1px dashed var(--rule);background:#faf8f3}
.pbody{padding:9px 12px;font-family:"SF Mono",ui-monospace,Menlo,monospace;font-size:12px;line-height:1.55;white-space:pre-wrap;word-break:break-word;color:#33312a}
.pf{font-weight:700;border-radius:3px;padding:0 3px}
.pf-base{background:#ececec;color:#333}.pf-force{background:#fbeae7;color:var(--no)}.pf-placebo{background:#fcf3e0;color:var(--amber)}
.condblock{border-left:4px solid var(--rule);border-radius:0 8px 8px 0;padding:2px 0 8px 14px;margin:16px 0;background:linear-gradient(90deg,rgba(0,0,0,.015),transparent 40%)}
.cb-base{border-left-color:#b9b4a8} .cb-force{border-left-color:var(--go)} .cb-placebo{border-left-color:var(--amber)}
.cb-base .cond{color:#5a564e} .cb-force .cond{color:var(--go)} .cb-placebo .cond{color:var(--amber)}
.hide2{display:none}
</style></head><body>
<header><span class=title>Read-back log viewer</span><span class=run id=run></span>
  <span class=kpi id=kpi></span><span class=pill id=verdict></span></header>
<div class=wrap>
  <div class=list id=list></div>
  <div class=detail id=detail><div class=empty>Select a prompt on the left.</div></div>
</div>
<script>
const DATA = __PAYLOAD__, G = DATA.groups, ST = DATA.stats||{};
document.getElementById('run').textContent = DATA.run;
const f=(c)=>(c&&c[0]!=null)?c[0].toFixed(3):'-';
document.getElementById('kpi').innerHTML =
 `KL force <b>${f(ST.mean_kl_force)}</b> &nbsp; KL placebo <b>${f(ST.mean_kl_placebo)}</b> &nbsp; p <b>${ST.perm_p!=null?ST.perm_p.toFixed(3):'-'}</b> &nbsp; prompts <b>${G.length}</b>`;
const vp=document.getElementById('verdict');
if(ST.detected!==undefined){vp.textContent=ST.detected?'read-back DETECTED':'not detected';vp.className='pill '+(ST.detected?'det':'no');}
function esc(s){return (s||'').replace(/[&<>]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[m]))}
function sentSplit(t){t=(t||'').replace(/\s+/g,' ').trim();return t.split(/(?<=[.!?])\s+(?=[A-Z"'(])/).filter(Boolean);}
function renderPrefill(text,p,cls){
  const ss=sentSplit(text); if(p-1>=ss.length) return esc(text);
  return ss.map((s,k)=>{ if(k!==p-1) return esc(s);
    const m=s.match(/[A-Za-z]/); if(!m) return esc(s); const i=s.indexOf(m[0]);
    return esc(s.slice(0,i))+`<span class="pf ${cls}">`+esc(s.slice(i,i+1))+'</span>'+esc(s.slice(i+1));
  }).join(' ');
}
function hlFirst(text,outcome){const e=esc(text);const m=e.match(/[A-Za-z]/);if(!m)return e;const i=e.indexOf(m[0]);
  return e.slice(0,i)+`<span class="firstlet fl-${outcome}">`+e.slice(i,i+1)+'</span>'+e.slice(i+1);}
function pct(g){const c=g.fcount||{};const n=(c.readback||0)+(c.original||0)+(c.offchain||0)||1;
  return {rb:100*(c.readback||0)/n, og:100*(c.original||0)/n, off:100*(c.offchain||0)/n};}

document.getElementById('list').innerHTML = G.map((g,i)=>{
  const p=pct(g); const c=g.fcount||{};
  return `<div class=row data-i=${i}>
    <div class=t>T${g.trial} · p${g.p} · forced ${g.orig}→${g.forced}</div>
    <div class=q>${esc(g.q)}</div>
    <div class=minibar><i class=rb style="width:${p.rb}%"></i><i class=og style="width:${p.og}%"></i><i class=off style="width:${p.off}%"></i></div>
    <div class=leg>forced: read-back ${c.readback||0} · original ${c.original||0} · off ${c.offchain||0}</div>
  </div>`;}).join('');

function condBlock(g, cond){
  const samps = (g.samples||{})[cond]||[]; if(!samps.length) return '';
  const order = cond==='force' ? ['readback','original','offchain'] : ['onchain','offchain'];
  const labels = {readback:`read-back (→ '${g.rb_pred}')`, original:`original (→ '${g.orig_next}', ignored the force)`,
                  offchain:'off-chain (abandoned the game)', onchain:`on-chain (→ '${g.orig_next}')`};
  const byo = {}; samps.forEach(s=>{(byo[s.outcome]=byo[s.outcome]||[]).push(s);});
  const badge = {base:'untouched',force:'forced',placebo:'placebo'}[cond];
  const pf = (g.prefills||{})[cond] || '';
  const noteBy = {base:'original sentence kept', force:'sentence forced to WRONG letter', placebo:'reworded, SAME first letter'};
  let html = `<div class=cond>${cond.toUpperCase()}<span class="badge b-${cond}">${badge}</span></div>`;
  if(pf) html += `<div class=prefill><div class=plabel>assistant · prefill = answer so far (sentence ${g.p} ${noteBy[cond]})</div>
    <div class=pbody>${renderPrefill(pf, g.p, 'pf-'+cond)}</div></div>`;
  order.forEach(o=>{const arr=byo[o]; if(!arr||!arr.length) return;
    html += `<div class=ogrp><div class=ohead onclick="this.parentNode.querySelector('.genwrap').classList.toggle('hide2')">
      <span class="dot ${o}"></span><span>${labels[o]}</span><span class=cnt>${arr.length}×</span></div>
      <div class=genwrap>`+arr.map(s=>`<div class=gen>${hlFirst(s.cont,o)}</div>`).join('')+`</div></div>`;});
  return `<div class="condblock cb-${cond}">`+html+`</div>`;
}
function show(i){
  const g=G[i]; document.querySelectorAll('.row').forEach(r=>r.classList.toggle('sel',+r.dataset.i===i));
  const seq=(g.expected||[]).map((L,idx)=>`<span class="seqL ${idx+1===g.p?'cur':''} ${idx+1===g.p+1?'nxt':''}">${idx+1}:${L}</span>`).join('<span class=arr>→</span>');
  document.getElementById('detail').innerHTML = `
    <div class=dh>${esc(g.q)}</div>
    <div class=sub>Trace <b>T${g.trial}</b>, we edit sentence <b>p${g.p}</b> (correct letter <b>${g.orig}</b>) and read sentence <b>p${g.p+1}</b>.
      Forcing it to <b style="color:var(--no)">${g.forced}</b> → read-back predicts <b style="color:var(--go)">${g.rb_pred}</b>; ignoring the force stays <b style="color:var(--accent)">${g.orig_next}</b>.</div>
    <div class=seqbar>${seq}<span class=seqcap>expected sequence (red = edited p${g.p}, green = read p${g.p+1})</span></div>
    <div class="msg user"><div class=role onclick="this.parentNode.querySelector('.body').classList.toggle('hide')"><span>user · prompt (click to toggle)</span></div>
      <div class="body hide">${esc(g.base_prompt)}</div></div>
    ${condBlock(g,'base')}
    ${condBlock(g,'force')}
    ${condBlock(g,'placebo')}`;
}
document.querySelectorAll('.row').forEach(r=> r.onclick=()=>show(+r.dataset.i));
</script></body></html>"""

out = run/'viewer.html'
out.write_text(HTML.replace('__PAYLOAD__', payload))
print('wrote', out, f'({len(groups)} prompts, {len(raws)} generations)')
