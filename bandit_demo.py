"""
DRL Assignment 1 - Group 159
Paper: "A Survey on Practical Applications of Multi-Armed and Contextual Bandits"
       Bouneffouf & Rish (2019), arXiv:1904.10040

The paper is a survey and runs no experiments of its own. This script is OUR OWN small
simulation of the core algorithms the survey describes, to justify why they are used and
to show how they behave:

  Part A  context-free MAB : epsilon-greedy vs UCB1 vs Thompson Sampling
  Part B  contextual bandit : context-free UCB1 vs LinUCB vs Linear Thompson Sampling

Run:  python bandit_demo.py        (needs only numpy; ~1 minute)
Output: results.json (cumulative pseudo-regret curves + final values)
Note: ~2 minutes on a laptop (Part B loops over arms in pure Python)
Everything is seeded, so the numbers on the slides are reproducible.
"""
import json
import numpy as np

SEED = 159  # group number


# --------------------------------------------------------------------------- Part A
def run_mab(algo, means, T, rng, eps=0.1):
    """One run of a K-armed Bernoulli bandit. Returns cumulative pseudo-regret."""
    K = len(means)
    best = means.max()
    n = np.zeros(K)          # pulls per arm
    s = np.zeros(K)          # summed reward per arm
    regret = np.zeros(T)
    total = 0.0
    for t in range(T):
        if t < K:                                   # play each arm once
            a = t
        elif algo == "eps_greedy":                  # explore w.p. eps, else exploit
            a = rng.integers(K) if rng.random() < eps else int(np.argmax(s / n))
        elif algo == "ucb1":                        # mean + sqrt(2 ln t / n)
            a = int(np.argmax(s / n + np.sqrt(2 * np.log(t + 1) / n)))
        elif algo == "thompson":                    # sample Beta posterior per arm
            a = int(np.argmax(rng.beta(1 + s, 1 + n - s)))
        r = float(rng.random() < means[a])          # Bernoulli reward
        n[a] += 1
        s[a] += r
        total += best - means[a]
        regret[t] = total
    return regret


# --------------------------------------------------------------------------- Part B
def run_cmab(algo, theta, T, rng, alpha=0.5, sigma=0.1):
    """One run of a K-armed linear contextual bandit, reward = theta_a . x + noise."""
    K, d = theta.shape
    A = np.stack([np.eye(d) for _ in range(K)])     # ridge matrices per arm
    b = np.zeros((K, d))
    n = np.zeros(K)
    s = np.zeros(K)
    regret = np.zeros(T)
    total = 0.0
    for t in range(T):
        x = rng.normal(size=d)
        x /= np.linalg.norm(x)                      # context on the unit sphere
        means = theta @ x
        if algo == "ucb1_nocontext":                # ignores x completely
            if t < K:
                a = t
            else:
                a = int(np.argmax(s / n + np.sqrt(2 * np.log(t + 1) / n)))
        elif algo == "linucb":                      # theta_hat.x + alpha*sqrt(x'A^-1 x)
            sc = np.empty(K)
            for k in range(K):
                Ainv = np.linalg.inv(A[k])
                sc[k] = (Ainv @ b[k]) @ x + alpha * np.sqrt(x @ Ainv @ x)
            a = int(np.argmax(sc))
        elif algo == "lin_ts":                      # sample theta from Gaussian posterior
            sc = np.empty(K)
            for k in range(K):
                Ainv = np.linalg.inv(A[k])
                th = rng.multivariate_normal(Ainv @ b[k], (alpha ** 2) * Ainv)
                sc[k] = th @ x
            a = int(np.argmax(sc))
        r = means[a] + sigma * rng.normal()
        A[a] += np.outer(x, x)
        b[a] += r * x
        n[a] += 1
        s[a] += r
        total += means.max() - means[a]
        regret[t] = total
    return regret


def main():
    out = {"seed": SEED}

    # ---- Part A: 10-armed Bernoulli bandit, fixed means, 50 runs, T = 20000 (long horizon to expose asymptotics)
    means = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55])
    T_A, RUNS_A = 20000, 50
    out["A"] = {"means": means.tolist(), "T": T_A, "runs": RUNS_A, "curves": {}, "final": {}}
    for algo in ["eps_greedy", "ucb1", "thompson"]:
        rng = np.random.default_rng(SEED)
        curves = np.mean([run_mab(algo, means, T_A, rng) for _ in range(RUNS_A)], axis=0)
        out["A"]["curves"][algo] = curves[399::400].round(2).tolist()   # steps 400, 800, ... 20000
        out["A"]["final"][algo] = round(float(curves[-1]), 1)
        print(f"[A] {algo:11s} final cumulative regret = {curves[-1]:.1f}")

    # ---- Part B: 5 arms, 5-d contexts, 50 runs, T = 3000
    K, d, T_B, RUNS_B = 5, 5, 3000, 50
    out["B"] = {"K": K, "d": d, "T": T_B, "runs": RUNS_B, "curves": {}, "final": {}}
    for algo in ["ucb1_nocontext", "linucb", "lin_ts"]:
        rng = np.random.default_rng(SEED)
        curves = []
        for _ in range(RUNS_B):
            theta = rng.normal(size=(K, d))
            theta /= np.linalg.norm(theta, axis=1, keepdims=True)
            curves.append(run_cmab(algo, theta, T_B, rng))
        curves = np.mean(curves, axis=0)
        out["B"]["curves"][algo] = curves[99::100].round(2).tolist()    # steps 100, 200, ... 3000
        out["B"]["final"][algo] = round(float(curves[-1]), 1)
        print(f"[B] {algo:14s} final cumulative regret = {curves[-1]:.1f}")

    with open("results.json", "w") as f:
        json.dump(out, f, indent=1)
    print("saved results.json")


if __name__ == "__main__":
    main()
