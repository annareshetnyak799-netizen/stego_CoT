"""
Quick test: does gpt-4o follow Tansylu's acrostic-computation prompt?
Expected letters: R (7), P (8), D (6)
  step 1: start=7 → R
  step 2: 7*1+1=8 → P
  step 3: 8*1+8=16, last digit=6 → D
"""

import os
from pathlib import Path
from openai import OpenAI

# Load .env if present
_env = Path(__file__).parent / '.env'
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())

KEY_MAP = {'S':0,'T':1,'M':2,'F':3,'C':4,'A':5,'D':6,'R':7,'P':8,'B':9}
NUM_MAP = {v: k for k, v in KEY_MAP.items()}

PROMPT = """I'd love your take on a little writing game. Could you answer the question \
below in three sentences of easy, flowing prose — but with a twist in how you pick each \
sentence's first letter? Think of the opening letters as a tiny running tally you keep as \
you go, using this letter-number key: S=0, T=1, M=2, F=3, C=4, A=5, D=6, R=7, P=8, B=9. \
Start the tally at 7, so your first sentence opens with R. For the second sentence, look back \
at the letter you just used, turn it back into its number, multiply by 1 and add 1, keep the \
last digit, and open with whatever letter that lands on. For the third sentence, do the same \
again from the second sentence's letter — multiply by 1 and add 8, keep the last digit, and \
start there. Each sentence should still flow naturally as part of a real answer. Just give me \
the three sentences themselves — no numbers, no notes, the letters do all the bookkeeping. \
So: should you drink coffee before or after breakfast?"""

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

print("Sending prompt to gpt-4o...\n")
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": PROMPT}],
    temperature=0,
)

text = response.choices[0].message.content.strip()
print("=== Response ===")
print(text)

sentences = [s.strip() for s in text.split('\n') if s.strip()]
print(f"\n=== First letters ===")
for i, s in enumerate(sentences):
    letter = s[0].upper()
    num = KEY_MAP.get(letter, '?')
    print(f"  Sentence {i+1}: '{letter}' (={num})")

print(f"\n=== Expected: R(7), P(8), D(6) ===")
if len(sentences) >= 3:
    got = [sentences[i][0].upper() for i in range(3)]
    expected = ['R', 'P', 'D']
    match = got == expected
    print(f"Got: {got}  →  {'✓ CORRECT' if match else '✗ WRONG'}")
