def run_kernel(indptr, nbr, tie, S0, T, n_warm, n_meas, seed):
    np.random.seed(seed)
    N = indptr.shape[0] - 1
    s = np.empty((2, N), np.int64)
    for r in range(2):
        for i in range(N):
            s[r, i] = S0[r, i]
    ref = np.empty((2, N), np.int64)
    out = np.zeros(9)   # |m|, m, m2, |q|, q2, persistence, flip rate, unsatisfied, count
    nflip = 0.0; nupd = 0.0
    for t in range(n_warm + n_meas):
        for r in range(2):
            for _u in range(N):
                i = np.random.randint(N)
                old = s[r, i]
                c = 0
                for q in range(indptr[i], indptr[i + 1]):
                    if tie[q] > 0:
                        c += 125 if s[r, nbr[q]] > 0 else 25
                    else:
                        c += 5 if s[r, nbr[q]] > 0 else 1
                c = c * 2 + (1 if old > 0 else 0)
                new = 1 if np.random.random() < T[c] else -1
                s[r, i] = new
                if t >= n_warm:
                    nupd += 1.0
                    if new != old:
                        nflip += 1.0
        if t == n_warm - 1:
            for r in range(2):
                for i in range(N):
                    ref[r, i] = s[r, i]
        if t >= n_warm:
            q = 0.0; m1 = 0.0; m2 = 0.0; c1 = 0.0; c2 = 0.0; un = 0.0
            for i in range(N):
                q += s[0, i] * s[1, i]; m1 += s[0, i]; m2 += s[1, i]
                c1 += s[0, i] * ref[0, i]; c2 += s[1, i] * ref[1, i]
                for e in range(indptr[i], indptr[i + 1]):
                    for r in range(2):
                        agree = s[r, i] == s[r, nbr[e]]
                        if (tie[e] > 0 and not agree) or (tie[e] < 0 and agree):
                            un += 0.5
            q /= N; m1 /= N; m2 /= N
            out[0] += 0.5 * (abs(m1) + abs(m2)); out[1] += 0.5 * (m1 + m2); out[2] += 0.5 * (m1 * m1 + m2 * m2)
            out[3] += abs(q); out[4] += q * q; out[5] += 0.5 * (c1 + c2) / N
            out[7] += un / max(indptr[N], 1); out[8] += 1
    res = out[:8] / out[8]
    res[6] = nflip / max(nupd, 1.0)
    return res
