import random
import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    # Fallback: keep API runnable even without numba.
    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco


@njit(cache=True)
def _simulate_abm_numba(T, c, m, n, d, k, R, s_0, z, init_node_ids, seed):
    np.random.seed(seed)

    init_n = len(init_node_ids)
    cap = max(64, init_n * 2 + 16)

    node_ids = np.empty(cap, dtype=np.int64)
    stress = np.empty(cap, dtype=np.float64)
    dead_flags = np.zeros(cap, dtype=np.uint8)

    for i in range(init_n):
        node_ids[i] = init_node_ids[i]
        stress[i] = s_0

    size = init_n

    S_ts = np.empty(T + 1, dtype=np.float64)
    P_ts = np.empty(T + 1, dtype=np.int64)
    birth_ts = np.zeros(T + 1, dtype=np.int64)
    death_ts = np.zeros(T + 1, dtype=np.int64)

    node_pop_ts = np.zeros((z, T + 1), dtype=np.float64)
    node_stress_ts = np.zeros((z, T + 1), dtype=np.float64)

    node_counts = np.zeros(z, dtype=np.int64)
    node_stress_sums = np.zeros(z, dtype=np.float64)

    r = R / z if z > 0 else 1.0
    r_safe = r if r > 0 else 1.0

    for t in range(T + 1):
        for ni in range(z):
            node_counts[ni] = 0
            node_stress_sums[ni] = 0.0

        total_stress = 0.0
        for i in range(size):
            ni = node_ids[i] - 1
            node_counts[ni] += 1
            s = stress[i]
            node_stress_sums[ni] += s
            total_stress += s

        P_ts[t] = size
        S_ts[t] = total_stress / size if size > 0 else 0.0

        for ni in range(z):
            pop = node_counts[ni]
            node_pop_ts[ni, t] = pop
            node_stress_ts[ni, t] = (node_stress_sums[ni] / pop) if pop > 0 else 0.0

        if t == T:
            break

        for i in range(size):
            dead_flags[i] = 0

        birth = 0
        death = 0

        idx = 0
        while idx < size:
            ni = node_ids[idx] - 1
            neighbors = node_counts[ni]

            s = stress[idx]
            A_i = c * (neighbors / r_safe) * (1.0 + s)
            s = (s + A_i) * (1.0 - d)
            if s < 0.0:
                s = 0.0
            stress[idx] = s

            if np.random.random() < (n / (1.0 + (s * k))):
                if size >= cap:
                    new_cap = cap * 2
                    new_node_ids = np.empty(new_cap, dtype=np.int64)
                    new_stress = np.empty(new_cap, dtype=np.float64)
                    new_dead = np.zeros(new_cap, dtype=np.uint8)

                    for j in range(size):
                        new_node_ids[j] = node_ids[j]
                        new_stress[j] = stress[j]
                        new_dead[j] = dead_flags[j]

                    node_ids = new_node_ids
                    stress = new_stress
                    dead_flags = new_dead
                    cap = new_cap

                node_ids[size] = node_ids[idx]
                stress[size] = s
                dead_flags[size] = 0
                size += 1
                birth += 1

            if np.random.random() < m:
                dead_flags[idx] = 1
                death += 1

            idx += 1

        if death > 0:
            write = 0
            for read in range(size):
                if dead_flags[read] == 0:
                    node_ids[write] = node_ids[read]
                    stress[write] = stress[read]
                    write += 1
            size = write

        birth_ts[t + 1] = birth
        death_ts[t + 1] = death

    final_node_ids = np.empty(size, dtype=np.int64)
    final_stress = np.empty(size, dtype=np.float64)
    for i in range(size):
        final_node_ids[i] = node_ids[i]
        final_stress[i] = stress[i]

    return S_ts, P_ts, birth_ts, death_ts, node_pop_ts, node_stress_ts, final_node_ids, final_stress


def _simulate_abm_python(T, c, m, n, d, k, R, s_0, z, init_node_ids, seed):
    np.random.seed(seed)

    init_n = len(init_node_ids)
    node_ids = np.array(init_node_ids, dtype=np.int64).copy()
    stress = np.full(init_n, s_0, dtype=np.float64)

    S_ts = np.empty(T + 1, dtype=np.float64)
    P_ts = np.empty(T + 1, dtype=np.int64)
    birth_ts = np.zeros(T + 1, dtype=np.int64)
    death_ts = np.zeros(T + 1, dtype=np.int64)

    node_pop_ts = np.zeros((z, T + 1), dtype=np.float64)
    node_stress_ts = np.zeros((z, T + 1), dtype=np.float64)

    r = R / z if z > 0 else 1.0
    r_safe = r if r > 0 else 1.0

    for t in range(T + 1):
        size = len(node_ids)
        if size > 0:
            node_counts = np.bincount(node_ids - 1, minlength=z).astype(np.int64)
            node_stress_sums = np.bincount(node_ids - 1, weights=stress, minlength=z).astype(np.float64)
            S_ts[t] = float(stress.mean())
        else:
            node_counts = np.zeros(z, dtype=np.int64)
            node_stress_sums = np.zeros(z, dtype=np.float64)
            S_ts[t] = 0.0
        P_ts[t] = size

        node_pop_ts[:, t] = node_counts
        node_stress_ts[:, t] = np.divide(
            node_stress_sums,
            node_counts,
            out=np.zeros(z, dtype=np.float64),
            where=node_counts > 0,
        )

        if t == T:
            break

        size = len(node_ids)
        if size == 0:
            continue

        dead_flags = np.zeros(size, dtype=bool)
        birth = 0
        death = 0

        idx = 0
        while idx < len(node_ids):
            ni = node_ids[idx] - 1
            neighbors = node_counts[ni]

            s = stress[idx]
            A_i = c * (neighbors / r_safe) * (1.0 + s)
            s = (s + A_i) * (1.0 - d)
            if s < 0.0:
                s = 0.0
            stress[idx] = s

            if np.random.random() < (n / (1.0 + (s * k))):
                node_ids = np.append(node_ids, node_ids[idx])
                stress = np.append(stress, s)
                dead_flags = np.append(dead_flags, False)
                birth += 1

            if np.random.random() < m:
                dead_flags[idx] = True
                death += 1

            idx += 1

        if death > 0:
            keep = ~dead_flags
            node_ids = node_ids[keep]
            stress = stress[keep]

        birth_ts[t + 1] = birth
        death_ts[t + 1] = death

    return S_ts, P_ts, birth_ts, death_ts, node_pop_ts, node_stress_ts, node_ids, stress


class IndividualModel:
    def __init__(self, T, c, m, n, d, k, R, P_0, s_0, z, track_deaths=True, engine="numba"):
        self.T = int(T)
        self.c = float(c)
        self.m = float(m)
        self.n = float(n)
        self.d = float(d)
        self.k = float(k)
        self.R = float(R)
        self.P_0 = int(P_0)
        self.s_0 = float(s_0)
        self.z = int(z)
        self.track_deaths = bool(track_deaths)
        self.engine = str(engine).lower()

        if self.engine not in ("numba", "python"):
            raise ValueError("engine must be 'numba' or 'python'")
        if self.engine == "numba" and not NUMBA_AVAILABLE:
            self.engine = "python"

        self.S = 0.0
        self.P = 0
        self.t = 0
        self.birth = 0
        self.death = 0

        self.S_ts = []
        self.P_ts = []
        self.birth_ts = []
        self.death_ts = []

        self.node_pop_ts = None
        self.node_stress_ts = None

        self._init_node_ids = None
        self._final_node_ids = None
        self._final_stress = None
        self._seed = None

        # Kept for compatibility with existing external code.
        self.death_list = []
        self.agents_list = []
        self.node = {i: [] for i in range(1, self.z + 1)}

    def setup(self):
        self._init_node_ids = np.empty(self.P_0, dtype=np.int64)
        for i in range(self.P_0):
            self._init_node_ids[i] = random.randint(1, self.z)

        # Seed for numba RNG derived from numpy RNG state.
        self._seed = int(np.random.randint(0, 2_147_483_647))

        self.t = 0
        self.birth = 0
        self.death = 0

    def run(self):
        if self._init_node_ids is None:
            self.setup()

        if self.engine == "numba":
            (
                self.S_ts,
                self.P_ts,
                self.birth_ts,
                self.death_ts,
                self.node_pop_ts,
                self.node_stress_ts,
                self._final_node_ids,
                self._final_stress,
            ) = _simulate_abm_numba(
                self.T,
                self.c,
                self.m,
                self.n,
                self.d,
                self.k,
                self.R,
                self.s_0,
                self.z,
                self._init_node_ids,
                self._seed,
            )
        else:
            (
                self.S_ts,
                self.P_ts,
                self.birth_ts,
                self.death_ts,
                self.node_pop_ts,
                self.node_stress_ts,
                self._final_node_ids,
                self._final_stress,
            ) = _simulate_abm_python(
                self.T,
                self.c,
                self.m,
                self.n,
                self.d,
                self.k,
                self.R,
                self.s_0,
                self.z,
                self._init_node_ids,
                self._seed,
            )

        self.t = self.T
        self.S = float(self.S_ts[-1]) if len(self.S_ts) else 0.0
        self.P = int(self.P_ts[-1]) if len(self.P_ts) else 0
        self.birth = int(self.birth_ts[-1]) if len(self.birth_ts) else 0
        self.death = int(self.death_ts[-1]) if len(self.death_ts) else 0

        # Lightweight compatibility mirror for node occupancy at final time.
        self.update_nodes()

    def step(self, sync_nodes_end=False):
        raise RuntimeError("step() is not supported in the numba-based IndividualModel. Use setup() + run().")

    def update_nodes(self):
        if self.node_pop_ts is None:
            for key in self.node:
                self.node[key] = []
            return

        t = min(self.t, self.T)
        for node_id in range(1, self.z + 1):
            pop = int(self.node_pop_ts[node_id - 1, t])
            self.node[node_id] = [None] * pop

    def get_node_snapshot(self, t=None):
        if self.node_pop_ts is None or self.node_stress_ts is None:
            return None, None

        if t is None:
            t = self.t
        t = int(max(0, min(t, self.T)))

        pop = self.node_pop_ts[:, t].copy()
        stress = self.node_stress_ts[:, t].copy()
        return pop, stress