import sys, itertools
import numpy as np


def parse(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]
    n, m = map(int, lines[0].split())
    c = np.array(list(map(float, lines[1].split())))
    A = []
    b = []
    sgn = []
    for ln in lines[2:2 + m]:
        *coeff, sign, val = ln.split()
        A.append(list(map(float, coeff)))
        sgn.append(sign)
        b.append(float(val))
    return n, np.array(A, float), np.array(b, float), np.array(c, float), sgn


def enumerate_bfs(n, A, b, sgn, c):
    m = len(A)
    eqA = [A[i] for i in range(m)]
    eqb = [b[i] for i in range(m)]

    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        eqA.append(e)
        eqb.append(0.0)
    eqA = np.vstack(eqA)
    eqb = np.array(eqb)
    best = None
    best_x = None
    idxs = range(len(eqA))

    for combo in itertools.combinations(idxs, n):
        M = eqA[list(combo)]
        if np.linalg.matrix_rank(M) < n:
            continue
        try:
            x = np.linalg.solve(M, eqb[list(combo)])
        except np.linalg.LinAlgError:
            continue

        if np.any(x < -1e-9):
            continue
        feas = True
        for i in range(m):
            lhs = A[i] @ x
            if sgn[i] == '<=' and lhs > b[i] + 1e-8:
                feas = False
                break
            if sgn[i] == '>=' and lhs < b[i] - 1e-8:
                feas = False
                break
            if sgn[i] == '=' and abs(lhs - b[i]) > 1e-8:
                feas = False
                break
        if not feas:
            continue
        z = float(np.dot(c, x))
        if (best is None) or (z > best + 1e-9):
            best = z
            best_x = x
    return best_x, best


def solve(path):
    n, A, b, c, sgn = parse(path)
    x, z = enumerate_bfs(n, A, b, sgn, c)
    if x is None:
        print('нет допустимых решений или целевая неограничена')
    else:
        print('оптимальная точка:', np.array2string(x, precision=6, separator=' '))
        print('значение целевой функции:', round(z, 6))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('использование: python lp_vertices.py input.txt')
    else:
        solve(sys.argv[1])
