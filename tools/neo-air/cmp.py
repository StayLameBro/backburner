# cmp.py REF_PREFIX TEST_PREFIX: teacher-forced comparison (TEST was fed REF's tokens): top-1 agreement and KL(ref || test)
# per position over the full vocabulary, in f64. Needs numpy.
import os, sys, numpy as np
def load(p):
    tok = [int(x) for x in open(p + '.tok')]
    V = os.path.getsize(p + '.logits') // (4 * len(tok))   # n_vocab, from the dump's size
    lg = np.fromfile(p + '.logits', dtype=np.float32).reshape(-1, V).astype(np.float64)
    return lg, [int(x) for x in open(p + '.argmax')], tok
r, ra, rt = load(sys.argv[1]); t, ta, tt = load(sys.argv[2])
n = min(len(r), len(t))
assert tt[:n] == rt[:n], "test was not fed the reference tokens"
def logsm(x): m = x.max(1, keepdims=True); return x - m - np.log(np.exp(x - m).sum(1, keepdims=True))
lr, lt = logsm(r[:n]), logsm(t[:n])
kl = (np.exp(lr) * (lr - lt)).sum(1)            # KL(ref || test) per position
agree = [ra[i] == ta[i] for i in range(n)]
first = next((i for i in range(n) if not agree[i]), None)
print(f"positions {n}: top-1 agreement {100*np.mean(agree):.1f}% ({sum(agree)}/{n}), greedy diverges at "
      f"{'none' if first is None else first}, mean KL {kl.mean():.5f}, max KL {kl.max():.5f} (pos {kl.argmax()}), "
      f"median {np.median(kl):.5f}")
