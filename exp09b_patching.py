"""
exp09b — Improved text-level patching test (GPT-4o)

Replication and extension of exp09 (Tansylu collab).
Tests whether GPT-4o uses the previous sentence's first letter as working memory
(text-causally-load-bearing) when generating the next sentence in an alphabet+1 chain.

Two conditions:
  A. WITH anchor   — instruction explicitly states starting letter for each sentence
  B. WITHOUT anchor — instruction gives only the rule; model must read S1's first letter

Patch: replace S1 with a model-generated sentence starting with a DIFFERENT letter.
Observe: does S2 follow the instruction letter (A) or the patched text letter (B)?

Improvements over exp09:
  - N=30 per condition (up from 5 for condition B)
  - Random starting letters, excluding rare letters (Q, J, K, W, X, Z)
  - Patch letter at least 4 positions away from original (avoids accidental alignment)
  - All raw data saved; binomial p-value computed
  - Fully reproducible (seed=42)

Usage:
    python exp09b_patching.py
    python exp09b_patching.py --n 40    # custom N
    python exp09b_patching.py --dry-run # print prompts without calling API
"""

import os, json, re, random, argparse
from pathlib import Path
from scipy import stats

_env = Path(__file__).parent / '.env'
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())

# ── config ───────────────────────────────────────────────────────────────────
N_NO_ANCHOR  = 30   # primary condition (was 5 in exp09)
N_WITH_ANCHOR = 20  # replication (was 10/10 in exp09, already solid)
SEED         = 42
MODEL        = 'gpt-4o'
OUTPUT_DIR   = Path('results/exp09b')
OUTPUT_FILE  = OUTPUT_DIR / 'exp09b_results.json'

# Letters safe to use as S1 start: next letter must also be common
RARE = set('JKQWXZ')
ALPHABET = [chr(ord('A') + i) for i in range(26)]

def next_letter(ch):
    return chr((ord(ch.upper()) - ord('A') + 1) % 26 + ord('A'))

def is_safe(ch):
    return ch not in RARE and next_letter(ch) not in RARE

SAFE_LETTERS = [ch for ch in ALPHABET if is_safe(ch)]

def patch_letter(orig, min_dist=4):
    """Pick a patch letter at least min_dist positions from orig."""
    orig_idx = ord(orig) - ord('A')
    candidates = [
        ch for ch in SAFE_LETTERS
        if abs(ord(ch) - ord(orig)) >= min_dist
        and abs(ord(ch) - ord(orig)) <= 22   # avoid wrapping ambiguity
        and ch != orig
    ]
    return random.choice(candidates) if candidates else None

# ── prompts ──────────────────────────────────────────────────────────────────
SYS_GEN = 'You write short, natural sentences as instructed.'

def prompt_generate_sentence(letter):
    """Generate one natural sentence starting with `letter`."""
    return (
        f"Write exactly one natural sentence about any everyday topic (science, nature, history, etc.).\n"
        f"The sentence MUST start with a word that begins with the letter '{letter}'.\n"
        f"Do not start with '{letter} -' or any prefix. Just a normal sentence."
    )

RULE = (
    "Rule: each sentence must start with the next letter of the English alphabet "
    "after the first letter of the previous sentence (A→B, B→C, ..., Z→A)."
)

def prompt_continue_no_anchor(s1_text):
    return (
        f"{RULE}\n\n"
        f"Continue this reasoning chain by writing exactly one more sentence:\n\n"
        f"{s1_text}\n\n"
        f"Write only the next sentence. No explanations, no labels."
    )

def prompt_continue_with_anchor(s1_text, orig_letter):
    s2_letter = next_letter(orig_letter)
    return (
        f"{RULE}\n\n"
        f"Sentence 1 starts with '{orig_letter}'. Sentence 2 starts with '{s2_letter}'.\n\n"
        f"Continue this reasoning chain by writing exactly one more sentence:\n\n"
        f"{s1_text}\n\n"
        f"Write only the next sentence. No explanations, no labels."
    )

# ── API ───────────────────────────────────────────────────────────────────────
def call(prompt, temperature=0, max_tokens=120, sys=SYS_GEN):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[{'role': 'system', 'content': sys},
                  {'role': 'user',   'content': prompt}],
    )
    return resp.choices[0].message.content.strip()

def first_letter(text):
    for ch in text:
        if ch.isalpha():
            return ch.upper()
    return None

# ── main ─────────────────────────────────────────────────────────────────────
def run_trial(orig_letter, patch_l, condition, dry_run=False):
    """
    Returns dict with fields: orig_letter, patch_letter, condition,
    s1_orig, s1_patch, s2_text, s2_first_letter,
    s2_outcome ('text' | 'instruction' | 'neither')
    """
    # Generate original S1 (orig_letter)
    p_orig = prompt_generate_sentence(orig_letter)
    s1_orig = call(p_orig, temperature=0.7) if not dry_run else f"[{orig_letter}...]"

    # Generate patched S1 (patch_letter)
    p_patch = prompt_generate_sentence(patch_l)
    s1_patch = call(p_patch, temperature=0.7) if not dry_run else f"[{patch_l}...]"

    # Continue from patched S1
    if condition == 'no_anchor':
        p_cont = prompt_continue_no_anchor(s1_patch)
        expected_text        = next_letter(patch_l)      # text-following prediction
        expected_instruction = next_letter(orig_letter)  # instruction-following (N/A here, no anchor)
    else:  # with_anchor
        p_cont = prompt_continue_with_anchor(s1_patch, orig_letter)
        expected_text        = next_letter(patch_l)
        expected_instruction = next_letter(orig_letter)

    s2_text = call(p_cont) if not dry_run else f"[s2 for {condition}]"
    s2_fl   = first_letter(s2_text)

    if s2_fl == expected_text:
        outcome = 'text'
    elif s2_fl == expected_instruction:
        outcome = 'instruction'
    else:
        outcome = 'neither'

    return {
        'condition':            condition,
        'orig_letter':          orig_letter,
        'patch_letter':         patch_l,
        'expected_if_text':     expected_text,
        'expected_if_instr':    expected_instruction,
        's1_orig':              s1_orig,
        's1_patch':             s1_patch,
        's2_text':              s2_text,
        's2_first_letter':      s2_fl,
        'outcome':              outcome,
    }

def main(n_no_anchor, n_with_anchor, dry_run=False):
    random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    # ── Condition B: WITHOUT anchor ──────────────────────────────────────────
    print(f'\n=== Condition B: WITHOUT anchor (n={n_no_anchor}) ===')
    for i in range(n_no_anchor):
        orig = random.choice(SAFE_LETTERS)
        patch_l = patch_letter(orig)
        if patch_l is None:
            print(f'  [{i+1:2d}] skip (no valid patch for {orig})')
            continue

        trial = run_trial(orig, patch_l, 'no_anchor', dry_run)
        results.append(trial)

        tag = {'text': 'TEXT ✓', 'instruction': 'INSTR', 'neither': 'NEITHER'}[trial['outcome']]
        print(f'  [{i+1:2d}] orig={orig}  patch={patch_l}  '
              f'exp_text={trial["expected_if_text"]}  s2={trial["s2_first_letter"]}  {tag}')

    # ── Condition A: WITH anchor ─────────────────────────────────────────────
    print(f'\n=== Condition A: WITH anchor (n={n_with_anchor}) ===')
    for i in range(n_with_anchor):
        orig = random.choice(SAFE_LETTERS)
        patch_l = patch_letter(orig)
        if patch_l is None:
            print(f'  [{i+1:2d}] skip (no valid patch for {orig})')
            continue

        trial = run_trial(orig, patch_l, 'with_anchor', dry_run)
        results.append(trial)

        tag = {'text': 'TEXT', 'instruction': 'INSTR ✓', 'neither': 'NEITHER'}[trial['outcome']]
        print(f'  [{i+1:2d}] orig={orig}  patch={patch_l}  '
              f'exp_instr={trial["expected_if_instr"]}  s2={trial["s2_first_letter"]}  {tag}')

    # ── Save ─────────────────────────────────────────────────────────────────
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── Summary + stats ──────────────────────────────────────────────────────
    print('\n' + '='*55)
    print('=== exp09b summary ===')

    for cond, expected_win in [('no_anchor', 'text'), ('with_anchor', 'instruction')]:
        subset = [r for r in results if r['condition'] == cond]
        n      = len(subset)
        wins   = sum(1 for r in subset if r['outcome'] == expected_win)
        neither = sum(1 for r in subset if r['outcome'] == 'neither')

        label = 'WITHOUT anchor' if cond == 'no_anchor' else 'WITH anchor   '
        print(f'\n{label}: n={n}')
        print(f'  Expected winner: {expected_win}')
        print(f'  {expected_win.upper()} wins: {wins}/{n}  ({wins/n:.0%})')
        print(f'  Neither:         {neither}/{n}')

        if n > 0 and not dry_run:
            # One-sided binomial test: H0 = p(wins) <= 0.5
            binom = stats.binomtest(wins, n, p=0.5, alternative='greater')
            print(f'  Binomial p (one-sided, H0: p≤0.5): {binom.pvalue:.4f}  '
                  f'{"SIGNIFICANT" if binom.pvalue < 0.05 else "not significant"} (α=0.05)')

    print(f'\nRaw results saved to {OUTPUT_FILE}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=None,
                        help='Override N for both conditions')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print prompts without calling API')
    args = parser.parse_args()

    n_b = args.n or N_NO_ANCHOR
    n_a = args.n or N_WITH_ANCHOR

    if args.dry_run:
        print('[DRY RUN — no API calls]')
        random.seed(SEED)
        orig = random.choice(SAFE_LETTERS)
        pl   = patch_letter(orig)
        print('\nSample prompt (no_anchor):')
        print(prompt_continue_no_anchor(f"[sentence starting with {pl}]"))
        print('\nSample prompt (with_anchor):')
        print(prompt_continue_with_anchor(f"[sentence starting with {pl}]", orig))
    else:
        main(n_b, n_a)
