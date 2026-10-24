import numpy as np

from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def _class_counts(y):
    y = np.asarray(y).astype(int)
    labels, counts = np.unique(y, return_counts=True)
    if len(labels) != 2:
        raise ValueError("QCSMOTEExact currently supports binary classification.")
    minority = labels[np.argmin(counts)]
    majority = labels[np.argmax(counts)]
    return minority, majority, dict(zip(labels, counts))


def _target_minority_count(y, sampling_strategy):
    minority, majority, counts = _class_counts(y)
    target = int(round(counts[majority] * float(sampling_strategy)))
    return minority, majority, counts, max(target, counts[minority])


def _safe_k(n_samples, k):
    return max(1, min(int(k), int(n_samples) - 1))


def _minmax01(values):
    values = np.asarray(values, dtype=float)
    lo = np.nanmin(values)
    hi = np.nanmax(values)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.ones_like(values, dtype=float)
    return (values - lo) / (hi - lo)


class QCSMOTEExact(BaseEstimator):
    """
    QC-SMOTE module-based implementation.

    Source:
    Upman and Gowda, "QC-SMOTE: Quality-Controlled SMOTE for
    Imbalanced Classification", arXiv:2606.24625, 2026.

    Important:
    This class follows Section 3.2-3.7 of the paper. The paper does not define
    q or p hyperparameters; its reported sensitivity defaults are k=5,
    best-of-K=3, overshoot factor=1.25, and purity threshold=0.3.

    Implemented modules:
    - local minority support S(x_i), minority density D(x_i), majority isolation R(x_i)
    - trustworthiness T(x_i) with normalized D and R
    - reliability-weighted synthetic allocation
    - IPQ best-of-K candidate selection
    - imbalance/overlap-adaptive Kc and interpolation range rho_c
    - acceptance check with adaptive local duplication fallback

    Thresholds and weights are exposed as constructor parameters because the
    paper describes them as hyperparameters or generic regime-policy constants.
    """

    def __init__(
        self,
        sampling_strategy=1.0,
        k_neighbors=5,
        best_of_k=3,
        overshoot_factor=1.0,
        purity_threshold=0.3,
        w_safe=1.0 / 3.0,
        w_density=1.0 / 3.0,
        w_isolation=1.0 / 3.0,
        eta=0.5,
        lambda_clearance=0.5,
        candidate_selection_mode="weighted_sum",
        purity_mode="uniform",
        purity_distance_power=1.0,
        purity_k_neighbors=None,
        minority_neighbor_scope="global",
        kmeans_n_clusters=8,
        kmeans_n_init=10,
        cluster_distance_gate_factor=1.05,
        cluster_outlier_fallback_quantile=1.0,
        tau1=0.30,
        tau2=0.60,
        k_low=None,
        k_mid=None,
        k_high=None,
        rho0=1.0,
        min_rho=0.05,
        theta_p=None,
        theta_c_factor=0.05,
        tau_c="median_majority_clearance",
        distance_metric="euclidean",
        distance_p=2,
        neighbor_quality_alpha=0.0,
        interpolation_distribution="uniform",
        interpolation_beta_a=1.0,
        interpolation_beta_b=1.0,
        allocation_smoothing_alpha=0.0,
        allocation_mode="multinomial",
        normalization_mode="minmax",
        normalization_q_low=0.10,
        normalization_q_high=0.90,
        isolation_mode="absolute_clearance",
        random_state=42,
        eps=1e-8,
    ):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.best_of_k = best_of_k
        self.overshoot_factor = overshoot_factor
        self.purity_threshold = purity_threshold
        self.w_safe = w_safe
        self.w_density = w_density
        self.w_isolation = w_isolation
        self.eta = eta
        self.lambda_clearance = lambda_clearance
        self.candidate_selection_mode = candidate_selection_mode
        self.purity_mode = purity_mode
        self.purity_distance_power = purity_distance_power
        self.purity_k_neighbors = purity_k_neighbors
        self.minority_neighbor_scope = minority_neighbor_scope
        self.kmeans_n_clusters = kmeans_n_clusters
        self.kmeans_n_init = kmeans_n_init
        self.cluster_distance_gate_factor = cluster_distance_gate_factor
        self.cluster_outlier_fallback_quantile = (
            cluster_outlier_fallback_quantile
        )
        self.tau1 = tau1
        self.tau2 = tau2
        self.k_low = k_low
        self.k_mid = k_mid
        self.k_high = k_high
        self.rho0 = rho0
        self.min_rho = min_rho
        self.theta_p = theta_p
        self.theta_c_factor = theta_c_factor
        self.tau_c = tau_c
        self.distance_metric = distance_metric
        self.distance_p = distance_p
        self.neighbor_quality_alpha = neighbor_quality_alpha
        self.interpolation_distribution = interpolation_distribution
        self.interpolation_beta_a = interpolation_beta_a
        self.interpolation_beta_b = interpolation_beta_b
        self.allocation_smoothing_alpha = allocation_smoothing_alpha
        self.allocation_mode = allocation_mode
        self.normalization_mode = normalization_mode
        self.normalization_q_low = normalization_q_low
        self.normalization_q_high = normalization_q_high
        self.isolation_mode = isolation_mode
        self.random_state = random_state
        self.eps = eps

    def _full_neighbours_without_self(self, X, query_indices, k):
        n_neighbors = min(len(X), k + 1)
        nn = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric=self.distance_metric,
            p=self.distance_p,
        ).fit(X)
        dists, idx = nn.kneighbors(X[query_indices], return_distance=True)

        out_dists = []
        out_idx = []
        for row_d, row_i, self_i in zip(dists, idx, query_indices):
            keep = row_i != self_i
            row_d = row_d[keep][:k]
            row_i = row_i[keep][:k]
            out_dists.append(row_d)
            out_idx.append(row_i)
        return out_dists, out_idx

    def _local_purity(self, X_all, y_all, x_query, minority, k):
        purity_k = k if self.purity_k_neighbors is None else int(self.purity_k_neighbors)
        if purity_k < 1:
            raise ValueError("purity_k_neighbors must be at least 1.")
        n_neighbors = min(len(X_all), purity_k)
        query = np.asarray(x_query, dtype=np.float32).reshape(1, -1)
        mode = str(self.purity_mode).lower()
        if mode == "uniform":
            idx = self.nn_full_.kneighbors(
                query, n_neighbors=n_neighbors, return_distance=False
            )[0]
            return float(np.mean(y_all[idx] == minority))
        if mode != "inverse_distance":
            raise ValueError(
                "purity_mode must be 'uniform' or 'inverse_distance'."
            )
        power = float(self.purity_distance_power)
        if power <= 0.0:
            raise ValueError("purity_distance_power must be positive.")
        distances, idx = self.nn_full_.kneighbors(
            query, n_neighbors=n_neighbors, return_distance=True
        )
        distances = distances[0]
        idx = idx[0]
        weights = 1.0 / np.power(distances + self.eps, power)
        return float(
            np.sum(weights * (y_all[idx] == minority)) / np.sum(weights)
        )

    def _majority_clearance(self, x_query):
        dist, _ = self.nn_majority_.kneighbors(np.asarray(x_query, dtype=np.float32).reshape(1, -1), n_neighbors=1, return_distance=True)
        return float(dist[0, 0])

    def _compute_trustworthiness(self, X, y, minority, majority, k):
        min_global_idx = np.flatnonzero(y == minority)
        full_dists, full_idx = self._full_neighbours_without_self(X, min_global_idx, k)

        S = np.zeros(len(min_global_idx), dtype=float)
        D = np.zeros(len(min_global_idx), dtype=float)
        R = np.zeros(len(min_global_idx), dtype=float)

        nearest_majority_dist, _ = self.nn_majority_.kneighbors(X[min_global_idx], n_neighbors=1, return_distance=True)
        nearest_majority_dist = nearest_majority_dist[:, 0]

        if len(min_global_idx) > 1:
            minority_k = _safe_k(len(min_global_idx), k)
            nn_minority = NearestNeighbors(
                n_neighbors=minority_k + 1,
                metric=self.distance_metric,
                p=self.distance_p,
            ).fit(X[min_global_idx])
            minority_dists = nn_minority.kneighbors(
                X[min_global_idx],
                return_distance=True,
            )[0][:, 1:]
            mean_minority_dist = np.mean(minority_dists, axis=1)
        else:
            mean_minority_dist = np.full(
                len(min_global_idx),
                self.eps,
                dtype=float,
            )

        for pos, (row_d, row_i) in enumerate(zip(full_dists, full_idx)):
            row_y = y[row_i]
            min_mask = row_y == minority
            maj_mask = row_y == majority

            S[pos] = float(np.mean(min_mask)) if len(row_y) else 0.0

            min_d = row_d[min_mask]
            if len(min_d) > 0:
                D[pos] = 1.0 / (self.eps + float(np.mean(min_d)))
            else:
                D[pos] = 0.0

            maj_d = row_d[maj_mask]
            if len(maj_d) > 0:
                R[pos] = float(np.mean(maj_d))
            else:
                R[pos] = float(nearest_majority_dist[pos])

        D_norm = self._normalize_quality_component(D)
        isolation_mode = str(self.isolation_mode).lower()
        if isolation_mode == "absolute_clearance":
            R_norm = self._normalize_quality_component(R)
        elif isolation_mode == "relative_margin":
            R_norm = nearest_majority_dist / (
                nearest_majority_dist + mean_minority_dist + self.eps
            )
            R_norm = np.clip(R_norm, 0.0, 1.0)
        else:
            raise ValueError(
                "isolation_mode must be 'absolute_clearance' "
                "or 'relative_margin'."
            )

        w_sum = float(self.w_safe + self.w_density + self.w_isolation)
        if w_sum <= 0:
            raise ValueError("Trustworthiness weights must sum to a positive value.")
        ws = self.w_safe / w_sum
        wd = self.w_density / w_sum
        wr = self.w_isolation / w_sum
        T = ws * S + wd * D_norm + wr * R_norm
        T = np.maximum(T, self.eps)

        return {
            "minority_indices": min_global_idx,
            "S": S,
            "D": D,
            "R": R,
            "D_norm": D_norm,
            "R_norm": R_norm,
            "mean_minority_distance": mean_minority_dist,
            "nearest_majority_distance": nearest_majority_dist,
            "T": T,
            "OV": 1.0 - float(np.mean(S)),
        }

    def _normalize_quality_component(self, values):
        mode = str(self.normalization_mode).lower()
        if mode == "minmax":
            return _minmax01(values)
        if mode != "quantile":
            raise ValueError(
                "normalization_mode must be 'minmax' or 'quantile'."
            )

        q_low = float(self.normalization_q_low)
        q_high = float(self.normalization_q_high)
        if not 0.0 <= q_low < q_high <= 1.0:
            raise ValueError(
                "normalization quantiles must satisfy "
                "0 <= q_low < q_high <= 1."
            )
        values = np.asarray(values, dtype=float)
        lo, hi = np.nanquantile(values, [q_low, q_high])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return _minmax01(values)
        return np.clip((values - lo) / (hi - lo), 0.0, 1.0)

    def _regime_params(self, overlap):
        k_mid = int(self.best_of_k if self.k_mid is None else self.k_mid)
        k_low = int(max(1, k_mid - 2) if self.k_low is None else self.k_low)
        k_high = int(k_mid + 2 if self.k_high is None else self.k_high)

        if overlap < self.tau1:
            kc = k_low
        elif overlap < self.tau2:
            kc = k_mid
        else:
            kc = k_high
        kc = max(1, kc)
        rho = float(self.rho0) * (1.0 - float(overlap))
        rho = min(1.0, max(float(self.min_rho), rho))
        return kc, rho

    def _tau_clearance(self, minority_clearances):
        if isinstance(self.tau_c, str):
            if self.tau_c != "median_majority_clearance":
                raise ValueError(f"Unknown tau_c mode: {self.tau_c}")
            tau = float(np.median(minority_clearances))
        else:
            tau = float(self.tau_c)
        return max(tau, self.eps)

    def _minority_neighbour_indices(self, X_min, k):
        if len(X_min) < 2:
            return [[] for _ in range(len(X_min))]
        k_eff = _safe_k(len(X_min), k)
        nn = NearestNeighbors(
            n_neighbors=k_eff + 1,
            metric=self.distance_metric,
            p=self.distance_p,
        ).fit(X_min)
        idx = nn.kneighbors(X_min, return_distance=False)[:, 1:]
        return [row.tolist() for row in idx]

    def _kmeans_minority_neighbour_indices(self, X, y, minority, X_min, k):
        n_clusters = int(self.kmeans_n_clusters)
        if n_clusters < 2 or n_clusters > len(X):
            raise ValueError(
                "kmeans_n_clusters must be between 2 and the number of "
                "training samples."
            )
        n_init = int(self.kmeans_n_init)
        if n_init < 1:
            raise ValueError("kmeans_n_init must be at least 1.")

        self.kmeans_model_ = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            random_state=self.random_state,
        ).fit(X)
        minority_clusters = self.kmeans_model_.labels_[y == minority]
        global_neighbours = self._minority_neighbour_indices(X_min, k)
        neighbours = [None] * len(X_min)
        fallback = np.zeros(len(X_min), dtype=bool)

        for cluster_id in range(n_clusters):
            positions = np.flatnonzero(minority_clusters == cluster_id)
            if len(positions) == 0:
                continue
            if len(positions) == 1:
                seed_pos = int(positions[0])
                neighbours[seed_pos] = global_neighbours[seed_pos]
                fallback[seed_pos] = True
                continue

            local_k = min(int(k), len(positions) - 1)
            nn = NearestNeighbors(
                n_neighbors=local_k + 1,
                metric=self.distance_metric,
                p=self.distance_p,
            ).fit(X_min[positions])
            local_indices = nn.kneighbors(
                X_min[positions], return_distance=False
            )[:, 1:]
            for row, seed_pos in zip(local_indices, positions):
                neighbours[int(seed_pos)] = positions[row].astype(int).tolist()

        for seed_pos, row in enumerate(neighbours):
            if row is None:
                neighbours[seed_pos] = global_neighbours[seed_pos]
                fallback[seed_pos] = True

        return neighbours, fallback, minority_clusters

    def _distance_gated_kmeans_neighbour_indices(
        self, X, y, minority, X_min, k
    ):
        gate_factor = float(self.cluster_distance_gate_factor)
        if gate_factor < 1.0:
            raise ValueError("cluster_distance_gate_factor must be at least 1.0.")
        n_clusters = int(self.kmeans_n_clusters)
        if n_clusters < 2 or n_clusters > len(X):
            raise ValueError(
                "kmeans_n_clusters must be between 2 and the number of "
                "training samples."
            )
        n_init = int(self.kmeans_n_init)
        if n_init < 1:
            raise ValueError("kmeans_n_init must be at least 1.")

        self.kmeans_model_ = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            random_state=self.random_state,
        ).fit(X)
        minority_clusters = self.kmeans_model_.labels_[y == minority]

        k_eff = _safe_k(len(X_min), k)
        global_nn = NearestNeighbors(
            n_neighbors=k_eff + 1,
            metric=self.distance_metric,
            p=self.distance_p,
        ).fit(X_min)
        global_distances, global_indices = global_nn.kneighbors(
            X_min, return_distance=True
        )
        global_distances = global_distances[:, 1:]
        global_indices = global_indices[:, 1:]

        neighbours = [row.astype(int).tolist() for row in global_indices]
        fallback = np.zeros(len(X_min), dtype=bool)
        replacement_counts = np.zeros(len(X_min), dtype=int)

        for cluster_id in range(n_clusters):
            positions = np.flatnonzero(minority_clusters == cluster_id)
            if len(positions) == 0:
                continue
            if len(positions) == 1:
                fallback[int(positions[0])] = True
                continue

            local_k = min(k_eff, len(positions) - 1)
            local_nn = NearestNeighbors(
                n_neighbors=local_k + 1,
                metric=self.distance_metric,
                p=self.distance_p,
            ).fit(X_min[positions])
            local_distances, local_indices = local_nn.kneighbors(
                X_min[positions], return_distance=True
            )
            local_distances = local_distances[:, 1:]
            local_indices = local_indices[:, 1:]

            for local_row, seed_pos in enumerate(positions):
                seed_pos = int(seed_pos)
                pool = neighbours[seed_pos]
                pool_set = set(pool)
                radius = gate_factor * float(global_distances[seed_pos, -1])
                extras = []
                for distance, candidate_local_pos in zip(
                    local_distances[local_row], local_indices[local_row]
                ):
                    candidate_pos = int(positions[int(candidate_local_pos)])
                    if (
                        candidate_pos not in pool_set
                        and float(distance) <= radius + self.eps
                    ):
                        extras.append(candidate_pos)

                cross_cluster_slots = [
                    slot
                    for slot, candidate_pos in enumerate(pool)
                    if minority_clusters[candidate_pos] != cluster_id
                ]
                cross_cluster_slots.sort(
                    key=lambda slot: global_distances[seed_pos, slot],
                    reverse=True,
                )
                for slot, candidate_pos in zip(cross_cluster_slots, extras):
                    pool[slot] = candidate_pos
                    replacement_counts[seed_pos] += 1

        return neighbours, fallback, minority_clusters, replacement_counts

    def _kmeans_outlier_fallback_neighbour_indices(
        self, X, y, minority, X_min, k
    ):
        quantile = float(self.cluster_outlier_fallback_quantile)
        if not 0.0 < quantile <= 1.0:
            raise ValueError(
                "cluster_outlier_fallback_quantile must be in (0, 1]."
            )

        hard_neighbours, singleton_fallback, minority_clusters = (
            self._kmeans_minority_neighbour_indices(
                X, y, minority, X_min, k
            )
        )
        k_eff = _safe_k(len(X_min), k)
        global_nn = NearestNeighbors(
            n_neighbors=k_eff + 1,
            metric=self.distance_metric,
            p=self.distance_p,
        ).fit(X_min)
        global_distances, global_indices = global_nn.kneighbors(
            X_min, return_distance=True
        )
        global_distances = global_distances[:, 1:]
        global_indices = global_indices[:, 1:]
        global_neighbours = [
            row.astype(int).tolist() for row in global_indices
        ]

        ratios = np.ones(len(X_min), dtype=float)
        p = float(self.distance_p)
        for seed_pos, pool in enumerate(hard_neighbours):
            if not pool:
                ratios[seed_pos] = np.inf
                continue
            differences = np.abs(
                X_min[np.asarray(pool, dtype=int)] - X_min[seed_pos]
            )
            if str(self.distance_metric).lower() == "minkowski":
                local_distances = np.sum(
                    np.power(differences, p), axis=1
                ) ** (1.0 / p)
            elif str(self.distance_metric).lower() == "manhattan":
                local_distances = np.sum(differences, axis=1)
            elif str(self.distance_metric).lower() == "euclidean":
                local_distances = np.sqrt(
                    np.sum(np.square(differences), axis=1)
                )
            else:
                raise ValueError(
                    "kmeans_cluster_outlier_fallback supports minkowski, "
                    "manhattan, or euclidean distance."
                )
            ratios[seed_pos] = float(
                np.max(local_distances)
                / max(float(global_distances[seed_pos, -1]), self.eps)
            )

        finite_ratios = ratios[np.isfinite(ratios)]
        threshold = float(np.quantile(finite_ratios, quantile))
        outlier_fallback = ratios > threshold
        fallback = singleton_fallback | outlier_fallback
        neighbours = [
            global_neighbours[seed_pos]
            if fallback[seed_pos]
            else hard_neighbours[seed_pos]
            for seed_pos in range(len(X_min))
        ]
        return (
            neighbours,
            fallback,
            minority_clusters,
            ratios,
            outlier_fallback,
            threshold,
        )

    def _fallback_sample(self, seed_pos, min_neigh, T, X_min):
        candidates = [seed_pos] + list(min_neigh[seed_pos])
        best_pos = max(candidates, key=lambda i: T[i])
        return X_min[best_pos].copy()

    def _sample_interpolation_lambda(self, rng, rho):
        mode = str(self.interpolation_distribution).lower()
        if mode == "uniform":
            fraction = rng.uniform(0.0, 1.0)
        elif mode == "beta":
            a = float(self.interpolation_beta_a)
            b = float(self.interpolation_beta_b)
            if a <= 0.0 or b <= 0.0:
                raise ValueError(
                    "interpolation_beta_a and interpolation_beta_b "
                    "must be positive."
                )
            fraction = rng.beta(a, b)
        else:
            raise ValueError(
                "interpolation_distribution must be 'uniform' or 'beta'."
            )
        return float(rho) * float(fraction)

    def fit_resample(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(int)
        rng = np.random.RandomState(self.random_state)

        target_strategy = float(self.sampling_strategy) * float(self.overshoot_factor)
        minority, majority, counts, target_min = _target_minority_count(y, target_strategy)
        n_new = target_min - counts[minority]
        if n_new <= 0:
            return X, y

        X_min = X[y == minority]
        X_maj = X[y == majority]
        if len(X_min) < 2 or len(X_maj) < 1:
            return X, y

        k = _safe_k(len(X), self.k_neighbors)
        self.nn_full_ = NearestNeighbors(
            n_neighbors=min(len(X), max(k, 1)),
            metric=self.distance_metric,
            p=self.distance_p,
        ).fit(X)
        self.nn_majority_ = NearestNeighbors(
            n_neighbors=1,
            metric=self.distance_metric,
            p=self.distance_p,
        ).fit(X_maj)

        stats = self._compute_trustworthiness(X, y, minority, majority, k)
        T = stats["T"]
        S = stats["S"]
        overlap = stats["OV"]
        ir = counts[majority] / max(counts[minority], 1)
        kc, rho = self._regime_params(overlap)

        minority_clearances, _ = self.nn_majority_.kneighbors(X_min, n_neighbors=1, return_distance=True)
        minority_clearances = minority_clearances[:, 0]
        tau_clearance = self._tau_clearance(minority_clearances)
        theta_c = float(self.theta_c_factor) * tau_clearance

        prob = T / np.sum(T)
        allocation_smoothing_alpha = float(self.allocation_smoothing_alpha)
        if not 0.0 <= allocation_smoothing_alpha <= 1.0:
            raise ValueError("allocation_smoothing_alpha must be in [0, 1].")
        if allocation_smoothing_alpha > 0.0:
            uniform_prob = np.full(len(T), 1.0 / len(T), dtype=float)
            prob = (
                (1.0 - allocation_smoothing_alpha) * prob
                + allocation_smoothing_alpha * uniform_prob
            )
        allocation_mode = str(self.allocation_mode).lower()
        if allocation_mode == "multinomial":
            allocations = rng.multinomial(n_new, prob)
        elif allocation_mode == "weighted_without_replacement":
            full_rounds, remainder = divmod(int(n_new), len(T))
            allocations = np.full(len(T), full_rounds, dtype=int)
            if remainder > 0:
                selected = rng.choice(
                    len(T),
                    size=remainder,
                    replace=False,
                    p=prob,
                )
                allocations[selected] += 1
        else:
            raise ValueError(
                "allocation_mode must be 'multinomial' or "
                "'weighted_without_replacement'."
            )

        minority_neighbor_scope = str(self.minority_neighbor_scope).lower()
        if minority_neighbor_scope == "global":
            min_neigh = self._minority_neighbour_indices(
                X_min, self.k_neighbors
            )
            cluster_fallback_seeds = np.zeros(len(X_min), dtype=bool)
            minority_cluster_labels = np.full(len(X_min), -1, dtype=int)
            kmeans_inertia = np.nan
            clusters_with_minority = 0
            singleton_minority_clusters = 0
            cluster_neighbor_replacement_counts = np.zeros(
                len(X_min), dtype=int
            )
            cluster_distance_ratios = np.full(len(X_min), np.nan)
            cluster_outlier_fallback_seeds = np.zeros(len(X_min), dtype=bool)
            cluster_outlier_ratio_threshold = np.nan
        elif minority_neighbor_scope == "kmeans_cluster":
            (
                min_neigh,
                cluster_fallback_seeds,
                minority_cluster_labels,
            ) = self._kmeans_minority_neighbour_indices(
                X, y, minority, X_min, self.k_neighbors
            )
            kmeans_inertia = float(self.kmeans_model_.inertia_)
            _, minority_cluster_counts = np.unique(
                minority_cluster_labels, return_counts=True
            )
            clusters_with_minority = int(len(minority_cluster_counts))
            singleton_minority_clusters = int(
                np.sum(minority_cluster_counts == 1)
            )
            cluster_neighbor_replacement_counts = np.zeros(
                len(X_min), dtype=int
            )
            cluster_distance_ratios = np.full(len(X_min), np.nan)
            cluster_outlier_fallback_seeds = np.zeros(len(X_min), dtype=bool)
            cluster_outlier_ratio_threshold = np.nan
        elif minority_neighbor_scope == "kmeans_distance_gated":
            (
                min_neigh,
                cluster_fallback_seeds,
                minority_cluster_labels,
                cluster_neighbor_replacement_counts,
            ) = self._distance_gated_kmeans_neighbour_indices(
                X, y, minority, X_min, self.k_neighbors
            )
            kmeans_inertia = float(self.kmeans_model_.inertia_)
            _, minority_cluster_counts = np.unique(
                minority_cluster_labels, return_counts=True
            )
            clusters_with_minority = int(len(minority_cluster_counts))
            singleton_minority_clusters = int(
                np.sum(minority_cluster_counts == 1)
            )
            cluster_distance_ratios = np.full(len(X_min), np.nan)
            cluster_outlier_fallback_seeds = np.zeros(len(X_min), dtype=bool)
            cluster_outlier_ratio_threshold = np.nan
        elif minority_neighbor_scope == "kmeans_cluster_outlier_fallback":
            (
                min_neigh,
                cluster_fallback_seeds,
                minority_cluster_labels,
                cluster_distance_ratios,
                cluster_outlier_fallback_seeds,
                cluster_outlier_ratio_threshold,
            ) = self._kmeans_outlier_fallback_neighbour_indices(
                X, y, minority, X_min, self.k_neighbors
            )
            cluster_neighbor_replacement_counts = np.zeros(
                len(X_min), dtype=int
            )
            kmeans_inertia = float(self.kmeans_model_.inertia_)
            _, minority_cluster_counts = np.unique(
                minority_cluster_labels, return_counts=True
            )
            clusters_with_minority = int(len(minority_cluster_counts))
            singleton_minority_clusters = int(
                np.sum(minority_cluster_counts == 1)
            )
        else:
            raise ValueError(
                "minority_neighbor_scope must be 'global', "
                "'kmeans_cluster', 'kmeans_distance_gated', or "
                "'kmeans_cluster_outlier_fallback'."
            )
        synth = []
        accepted_counts = np.zeros(len(T), dtype=int)
        fallback_counts = np.zeros(len(T), dtype=int)
        selected_quality_sum = np.zeros(len(T), dtype=float)
        selected_ipq_sum = np.zeros(len(T), dtype=float)
        selected_clearance_sum = np.zeros(len(T), dtype=float)
        neighbor_quality_alpha = float(self.neighbor_quality_alpha)
        if not 0.0 <= neighbor_quality_alpha <= 1.0:
            raise ValueError("neighbor_quality_alpha must be in [0, 1].")
        candidate_selection_mode = str(self.candidate_selection_mode).lower()
        if candidate_selection_mode not in {
            "weighted_sum",
            "ipq_then_clearance",
        }:
            raise ValueError(
                "candidate_selection_mode must be 'weighted_sum' or "
                "'ipq_then_clearance'."
            )

        for seed_pos, n_seed in enumerate(allocations):
            if n_seed <= 0:
                continue
            neighbours = min_neigh[seed_pos]
            if not neighbours:
                for _ in range(n_seed):
                    synth.append(X_min[seed_pos].copy())
                fallback_counts[seed_pos] += int(n_seed)
                continue

            selection_prob = None
            if neighbor_quality_alpha > 0.0:
                neighbour_quality = np.maximum(
                    T[np.asarray(neighbours, dtype=int)],
                    self.eps,
                )
                quality_prob = neighbour_quality / np.sum(neighbour_quality)
                uniform_prob = np.full(
                    len(neighbours),
                    1.0 / len(neighbours),
                    dtype=float,
                )
                selection_prob = (
                    (1.0 - neighbor_quality_alpha) * uniform_prob
                    + neighbor_quality_alpha * quality_prob
                )

            for _ in range(n_seed):
                if selection_prob is None:
                    nb_pos = rng.choice(neighbours)
                else:
                    nb_pos = rng.choice(neighbours, p=selection_prob)
                x_i = X_min[seed_pos]
                x_j = X_min[nb_pos]
                midpoint = (x_i + x_j) / 2.0
                midpoint_purity = self._local_purity(X, y, midpoint, minority, k)

                best = None
                best_q = -np.inf
                best_ipq = 0.0
                best_clearance = 0.0
                best_selection_key = None

                for _m in range(kc):
                    lam = self._sample_interpolation_lambda(rng, rho)
                    cand = x_i + lam * (x_j - x_i)
                    cand_purity = self._local_purity(X, y, cand, minority, k)
                    p_ipq = float(self.eta) * cand_purity + (1.0 - float(self.eta)) * midpoint_purity
                    clearance = self._majority_clearance(cand)
                    q = p_ipq + float(self.lambda_clearance) * min(1.0, clearance / tau_clearance)

                    if candidate_selection_mode == "weighted_sum":
                        selection_key = q
                    else:
                        selection_key = (p_ipq, clearance)

                    if best_selection_key is None or selection_key > best_selection_key:
                        best = cand
                        best_q = q
                        best_ipq = p_ipq
                        best_clearance = clearance
                        best_selection_key = selection_key

                theta_p = self.purity_threshold if self.theta_p is None else self.theta_p
                selected_quality_sum[seed_pos] += float(best_q)
                selected_ipq_sum[seed_pos] += float(best_ipq)
                selected_clearance_sum[seed_pos] += float(best_clearance)
                if best is not None and best_ipq >= float(theta_p) and best_clearance >= theta_c:
                    synth.append(best.astype(np.float32))
                    accepted_counts[seed_pos] += 1
                else:
                    synth.append(self._fallback_sample(seed_pos, min_neigh, T, X_min).astype(np.float32))
                    fallback_counts[seed_pos] += 1

        selected_counts = accepted_counts + fallback_counts
        selected_quality_mean = np.divide(
            selected_quality_sum,
            selected_counts,
            out=np.full(len(T), np.nan, dtype=float),
            where=selected_counts > 0,
        )
        selected_ipq_mean = np.divide(
            selected_ipq_sum,
            selected_counts,
            out=np.full(len(T), np.nan, dtype=float),
            where=selected_counts > 0,
        )
        selected_clearance_mean = np.divide(
            selected_clearance_sum,
            selected_counts,
            out=np.full(len(T), np.nan, dtype=float),
            where=selected_counts > 0,
        )
        self.diagnostics_ = {
            "minority_indices": stats["minority_indices"].copy(),
            "support": stats["S"].copy(),
            "density_raw": stats["D"].copy(),
            "isolation_raw": stats["R"].copy(),
            "density_norm": stats["D_norm"].copy(),
            "isolation_norm": stats["R_norm"].copy(),
            "mean_minority_distance": stats["mean_minority_distance"].copy(),
            "nearest_majority_distance": stats["nearest_majority_distance"].copy(),
            "trustworthiness": T.copy(),
            "allocation_probability": prob.copy(),
            "allocated_count": allocations.copy(),
            "accepted_count": accepted_counts,
            "fallback_count": fallback_counts,
            "selected_quality_mean": selected_quality_mean,
            "selected_ipq_mean": selected_ipq_mean,
            "selected_clearance_mean": selected_clearance_mean,
            "minority_cluster_label": minority_cluster_labels.copy(),
            "cluster_neighbor_fallback_seed": cluster_fallback_seeds.copy(),
            "cluster_neighbor_replacement_count": (
                cluster_neighbor_replacement_counts.copy()
            ),
            "cluster_distance_ratio": cluster_distance_ratios.copy(),
            "cluster_outlier_fallback_seed": (
                cluster_outlier_fallback_seeds.copy()
            ),
        }

        if not synth:
            return X, y

        X_new = np.vstack([X, np.asarray(synth, dtype=np.float32)])
        y_new = np.concatenate([y, np.full(len(synth), minority, dtype=y.dtype)])
        order = rng.permutation(len(y_new))

        self.stats_ = {
            "imbalance_ratio": float(ir),
            "overlap": float(overlap),
            "Kc": int(kc),
            "rho": float(rho),
            "overshoot_factor": float(self.overshoot_factor),
            "purity_threshold": float(self.purity_threshold if self.theta_p is None else self.theta_p),
            "tau_clearance": float(tau_clearance),
            "theta_c": float(theta_c),
            "mean_support": float(np.mean(S)),
            "mean_trustworthiness": float(np.mean(T)),
            "mean_isolation_score": float(np.mean(stats["R_norm"])),
            "n_requested_synthetic": int(n_new),
            "n_generated_synthetic": int(len(synth)),
            "n_accepted_synthetic": int(np.sum(accepted_counts)),
            "n_fallback_synthetic": int(np.sum(fallback_counts)),
            "acceptance_rate": float(np.sum(accepted_counts) / max(len(synth), 1)),
            "fallback_rate": float(np.sum(fallback_counts) / max(len(synth), 1)),
            "distance_metric": str(self.distance_metric),
            "distance_p": float(self.distance_p),
            "neighbor_quality_alpha": neighbor_quality_alpha,
            "candidate_selection_mode": candidate_selection_mode,
            "purity_mode": str(self.purity_mode),
            "purity_distance_power": float(self.purity_distance_power),
            "purity_k_neighbors": int(
                k if self.purity_k_neighbors is None else self.purity_k_neighbors
            ),
            "minority_neighbor_scope": minority_neighbor_scope,
            "kmeans_n_clusters": int(self.kmeans_n_clusters),
            "kmeans_n_init": int(self.kmeans_n_init),
            "cluster_distance_gate_factor": float(
                self.cluster_distance_gate_factor
            ),
            "cluster_outlier_fallback_quantile": float(
                self.cluster_outlier_fallback_quantile
            ),
            "cluster_outlier_ratio_threshold": float(
                cluster_outlier_ratio_threshold
            ),
            "kmeans_inertia": kmeans_inertia,
            "clusters_with_minority": clusters_with_minority,
            "singleton_minority_clusters": singleton_minority_clusters,
            "cluster_neighbor_fallback_seeds": int(
                np.sum(cluster_fallback_seeds)
            ),
            "cluster_neighbor_fallback_seed_rate": float(
                np.mean(cluster_fallback_seeds)
            ),
            "cluster_neighbor_fallback_allocation_rate": float(
                np.sum(allocations[cluster_fallback_seeds])
                / max(np.sum(allocations), 1)
            ),
            "cluster_neighbor_replacement_seeds": int(
                np.sum(cluster_neighbor_replacement_counts > 0)
            ),
            "cluster_neighbor_replacement_seed_rate": float(
                np.mean(cluster_neighbor_replacement_counts > 0)
            ),
            "cluster_neighbor_replacement_allocation_rate": float(
                np.sum(
                    allocations[cluster_neighbor_replacement_counts > 0]
                )
                / max(np.sum(allocations), 1)
            ),
            "mean_cluster_neighbor_replacements": float(
                np.mean(cluster_neighbor_replacement_counts)
            ),
            "cluster_outlier_fallback_seeds": int(
                np.sum(cluster_outlier_fallback_seeds)
            ),
            "cluster_outlier_fallback_seed_rate": float(
                np.mean(cluster_outlier_fallback_seeds)
            ),
            "cluster_outlier_fallback_allocation_rate": float(
                np.sum(allocations[cluster_outlier_fallback_seeds])
                / max(np.sum(allocations), 1)
            ),
            "mean_cluster_distance_ratio": float(
                np.nanmean(cluster_distance_ratios)
            )
            if np.any(np.isfinite(cluster_distance_ratios))
            else np.nan,
            "interpolation_distribution": str(
                self.interpolation_distribution
            ),
            "interpolation_beta_a": float(self.interpolation_beta_a),
            "interpolation_beta_b": float(self.interpolation_beta_b),
            "allocation_smoothing_alpha": allocation_smoothing_alpha,
            "allocation_mode": allocation_mode,
            "normalization_mode": str(self.normalization_mode),
            "normalization_q_low": float(self.normalization_q_low),
            "normalization_q_high": float(self.normalization_q_high),
            "isolation_mode": str(self.isolation_mode),
        }
        return X_new[order], y_new[order]
