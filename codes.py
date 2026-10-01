import pickle
import random
import math
import os
import numpy as np
from itertools import combinations
from scipy.optimize import linprog

# Input files. The program first tries to continue from current output files.
# If those do not exist, it falls back to the old provided files.
GENERATOR_INPUT_CANDIDATES = ["generatorMatrix", "generatorMatrix_old"]
MHEIGHT_INPUT_CANDIDATES = ["mHeight", "mHeight_old"]

GENERATOR_OUTPUT = "generatorMatrix"
MHEIGHT_OUTPUT = "mHeight"

# Parameter cases required by the project.
TARGET_PARAMS = [
    (9, 4, 4),
    (9, 6, 3),
    (9, 5, 4),
    (9, 5, 3),
    (9, 6, 2),
    (9, 4, 3),
    (9, 5, 2),
    (9, 4, 2),
    (9, 4, 5),
]


def find_file(candidates):
    # Return the first available file from a list of possible filenames.
    for name in candidates:
        if os.path.exists(name):
            return name
    raise FileNotFoundError("Could not find files: " + str(candidates))


def build_G(k, P):
    # Builds systematic generator matrix G = [I_k | P].
    return np.concatenate((np.eye(k), P.astype(float)), axis=1)


def compute_m_height(P, k, m):
    # Computes exact m-height using the LP-based method from the project.
    G = build_G(k, P)
    n = G.shape[1]
    indices = list(range(n))

    best = 0.0
    best_witness = None

    # Try every subset S of size m.
    for S_tuple in combinations(indices, m):
        S = set(S_tuple)
        Sc = [t for t in indices if t not in S]

        # For each j in S, solve one linear program.
        for j in S:
            c = -G[:, j]

            A = []
            b = []

            # Constraints: -1 <= G_t * u <= 1 for every t not in S.
            for t in Sc:
                A.append(G[:, t])
                b.append(1.0)
                A.append(-G[:, t])
                b.append(1.0)

            res = linprog(
                c=c,
                A_ub=np.array(A),
                b_ub=np.array(b),
                bounds=[(None, None)] * k,
                method="highs",
            )

            # If LP is unbounded, the m-height is infinite.
            if res.status == 3:
                return np.inf, (j, tuple(S))

            # Keep the largest LP objective value.
            if res.success:
                val = -res.fun
                if val > best:
                    best = val
                    best_witness = (j, tuple(S))

    return float(best), best_witness


def exact_m_height_lp_sj(n, k, m, P):
    # Wrapper function required by some testing setups.
    h, _ = compute_m_height(P, k, m)
    return float(h)


def repair_P(P):
    # Keeps entries in the allowed range [-100, 100] and fixes zero columns.
    P = np.clip(P, -100, 100).astype(int)

    for j in range(P.shape[1]):
        if np.all(P[:, j] == 0):
            P[random.randrange(P.shape[0]), j] = random.choice([-1, 1])

    return P


def random_matrix(k, cols, scale):
    # Creates a random integer P matrix with values from -scale to scale.
    P = np.random.randint(-scale, scale + 1, size=(k, cols))
    return repair_P(P)


def random_balanced_matrix(k, cols, scale):
    # Creates another type of random matrix to increase diversity.
    P = np.zeros((k, cols), dtype=int)

    for j in range(cols):
        values = np.random.randint(-scale, scale + 1, size=k)

        if np.all(values == 0):
            values[random.randrange(k)] = random.choice([-1, 1])

        P[:, j] = values

    return repair_P(P)


def matrix_key(P):
    # Converts matrix into a tuple so repeated matrices can be detected.
    return tuple(P.flatten().tolist())


def get_bad_cols(k, witness):
    # Uses the LP witness to identify columns involved in the worst m-height.
    if witness is None:
        return []

    j, S = witness
    bad_G_cols = set(S)
    bad_G_cols.add(j)

    # Convert G column indices to P column indices.
    return [col - k for col in bad_G_cols if col >= k]


def mutate_entry(P, bad_cols):
    # Changes one entry, preferably in a bad column.
    Q = P.copy()
    j = random.choice(bad_cols) if bad_cols else random.randrange(Q.shape[1])
    i = random.randrange(Q.shape[0])

    Q[i, j] += random.choice(
        [-15, -12, -10, -8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8, 10, 12, 15]
    )

    return repair_P(Q)


def mutate_column(P, bad_cols):
    # Changes several entries in one column.
    Q = P.copy()
    j = random.choice(bad_cols) if bad_cols else random.randrange(Q.shape[1])

    for i in range(Q.shape[0]):
        if random.random() < 0.75:
            Q[i, j] += random.choice(
                [-12, -10, -8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8, 10, 12]
            )

    return repair_P(Q)


def replace_column(P, bad_cols):
    # Replaces one column with a new random column.
    Q = P.copy()
    j = random.choice(bad_cols) if bad_cols else random.randrange(Q.shape[1])

    scale = random.choice([4, 6, 8, 10, 12, 16, 20, 25, 30, 40])
    new_col = np.random.randint(-scale, scale + 1, size=Q.shape[0])

    if np.all(new_col == 0):
        new_col[random.randrange(Q.shape[0])] = random.choice([-1, 1])

    Q[:, j] = new_col
    return repair_P(Q)


def replace_two_columns(P):
    # Replaces two columns to make a larger jump in the search space.
    Q = P.copy()
    cols = random.sample(range(Q.shape[1]), min(2, Q.shape[1]))

    for j in cols:
        scale = random.choice([4, 6, 8, 10, 12, 16, 20, 30])
        new_col = np.random.randint(-scale, scale + 1, size=Q.shape[0])

        if np.all(new_col == 0):
            new_col[random.randrange(Q.shape[0])] = random.choice([-1, 1])

        Q[:, j] = new_col

    return repair_P(Q)


def swap_columns(P):
    # Swaps two columns of P.
    Q = P.copy()

    if Q.shape[1] >= 2:
        j1, j2 = random.sample(range(Q.shape[1]), 2)
        Q[:, [j1, j2]] = Q[:, [j2, j1]]

    return repair_P(Q)


def random_restart_nearby(P):
    # Applies several random changes to escape local minima.
    Q = P.copy()

    for _ in range(random.randint(5, 14)):
        i = random.randrange(Q.shape[0])
        j = random.randrange(Q.shape[1])
        Q[i, j] += random.choice(
            [-20, -16, -12, -10, -8, -6, -4, -2, 2, 4, 6, 8, 10, 12, 16, 20]
        )

    return repair_P(Q)


def random_small_matrix_like(P):
    # Generates a completely fresh small random matrix with same dimensions.
    return random_matrix(P.shape[0], P.shape[1], random.choice([4, 6, 8, 10, 12, 16, 20]))


def generate_neighbor(P, k, witness):
    # Randomly chooses one mutation method to create a neighboring matrix.
    bad_cols = get_bad_cols(k, witness)
    move = random.random()

    if move < 0.25:
        return mutate_entry(P, bad_cols)

    if move < 0.50:
        return mutate_column(P, bad_cols)

    if move < 0.65:
        return replace_column(P, bad_cols)

    if move < 0.78:
        return replace_two_columns(P)

    if move < 0.88:
        return random_restart_nearby(P)

    if move < 0.97:
        return random_small_matrix_like(P)

    return swap_columns(P)


def search_settings(param):
    # Harder cases receive more rounds and a wider beam.
    if param in [(9, 4, 4), (9, 6, 3), (9, 5, 4), (9, 4, 5)]:
        return 120, 10, 14, 40

    if param in [(9, 5, 3), (9, 6, 2)]:
        return 90, 9, 12, 35

    return 70, 8, 10, 30


def accept_candidate(candidate_h, current_h, round_num, total_rounds):
    # Always accept better matrices. Sometimes accept worse ones early on
    # to help the search escape local minima.
    if candidate_h <= current_h:
        return True

    temperature = max(0.02, 0.40 * (1.0 - round_num / max(1, total_rounds)))
    gap = candidate_h - current_h

    probability = math.exp(-gap / max(temperature * max(1.0, current_h), 1e-9))
    return random.random() < probability


def improve_case(n, k, m, old_P, old_height):
    # Improves one specific (n, k, m) case.
    print("\n")
    print("Starting case:", (n, k, m))
    print("Old submitted height:", old_height)

    rounds, beam_width, trials_per_matrix, random_starts = search_settings((n, k, m))

    # Keep the old matrix as a safe fallback.
    old_P = repair_P(old_P)
    old_h, old_witness = compute_m_height(old_P, k, m)
    old_h = float(old_h)

    best_P = old_P.copy()
    best_h = old_h
    best_witness = old_witness

    # Generate random starting matrices instead of relying only on old_P.
    initial_matrices = []

    for _ in range(random_starts):
        scale = random.choice([3, 4, 5, 6, 8, 10, 12, 16, 20, 30, 40])
        initial_matrices.append(random_matrix(k, n - k, scale))

    for _ in range(random_starts):
        scale = random.choice([3, 4, 5, 6, 8, 10, 12, 16, 20])
        initial_matrices.append(random_balanced_matrix(k, n - k, scale))

    beam = []
    seen = set()

    # Evaluate starting matrices and keep the promising ones.
    for P0 in initial_matrices:
        key = matrix_key(P0)
        if key in seen:
            continue

        seen.add(key)

        h0, w0 = compute_m_height(P0, k, m)
        h0 = float(h0)

        if np.isinf(h0):
            continue

        beam.append((h0, P0, w0))

        if h0 < best_h:
            best_h = h0
            best_P = P0.copy()
            best_witness = w0
            print("New best from random initialization:", best_h)

    if len(beam) == 0:
        print("No valid random starting matrices. Keeping old result.")
        return old_P, old_h

    beam.sort(key=lambda item: item[0])
    beam = beam[:beam_width]

    print("Initial best height:", best_h)
    print("Settings: rounds =", rounds, "beam =", beam_width, "trials =", trials_per_matrix)

    # Main beam search loop.
    for r in range(rounds):
        candidates = []

        for current_h, current_P, current_witness in beam:
            for _ in range(trials_per_matrix):
                Q = generate_neighbor(current_P, k, current_witness)
                key = matrix_key(Q)

                if key in seen:
                    continue

                seen.add(key)

                h, witness = compute_m_height(Q, k, m)
                h = float(h)

                if np.isinf(h):
                    continue

                if accept_candidate(h, current_h, r, rounds):
                    candidates.append((h, Q, witness))

                if h < best_h:
                    best_h = h
                    best_P = Q.copy()
                    best_witness = witness
                    print("Improved", (n, k, m), "to", best_h)

        # Keep only the best candidates for the next round.
        all_options = beam + candidates
        all_options.append((best_h, best_P.copy(), best_witness))

        all_options.sort(key=lambda item: item[0])
        beam = all_options[:beam_width]

        if r % 5 == 0:
            print("Round", r, "best:", best_h, "beam best:", beam[0][0])

    # Only replace old result if the new matrix is strictly better.
    if best_h < old_h:
        print("Saved improved result:", best_h)
        return best_P, best_h

    print("No improvement. Kept old result:", old_h)
    return old_P, old_h


def run_once(input_generator, input_heights, output_generator, output_heights):
    # Runs the improvement procedure for all required cases once.
    with open(input_generator, "rb") as f:
        generator = pickle.load(f)

    with open(input_heights, "rb") as f:
        heights = pickle.load(f)

    new_generator = dict(generator)
    new_heights = dict(heights)

    for n, k, m in TARGET_PARAMS:
        P = new_generator[(n, k, m)]
        old_height = new_heights[(n, k, m)]

        best_P, best_h = improve_case(n, k, m, P, old_height)

        new_generator[(n, k, m)] = best_P
        new_heights[(n, k, m)] = best_h

        # Save progress after every case so work is not lost.
        with open(output_generator, "wb") as f:
            pickle.dump(new_generator, f)

        with open(output_heights, "wb") as f:
            pickle.dump(new_heights, f)

        print("Progress saved after", (n, k, m))

    return new_generator, new_heights


def run_search():
    # Runs several global restarts over all parameter cases.
    generator_input = find_file(GENERATOR_INPUT_CANDIDATES)
    mheight_input = find_file(MHEIGHT_INPUT_CANDIDATES)

    best_gen_file = generator_input
    best_h_file = mheight_input

    for restart in range(5):
        print("Restart: ", restart)

        run_once(best_gen_file, best_h_file, GENERATOR_OUTPUT, MHEIGHT_OUTPUT)

        best_gen_file = GENERATOR_OUTPUT
        best_h_file = MHEIGHT_OUTPUT

    print("\nDone.")
    print("Saved final:", GENERATOR_OUTPUT)
    print("Saved final:", MHEIGHT_OUTPUT)


def main():
    # Random seed is printed so a run can be repeated if needed.
    seed = random.randint(1, 1000000)
    print("Using seed:", seed)

    random.seed(seed)
    np.random.seed(seed)

    run_search()


if __name__ == "__main__":
    main()