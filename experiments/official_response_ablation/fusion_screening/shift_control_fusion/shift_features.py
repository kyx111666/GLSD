"""Exact, label-free common-time shift controls on the sealed physical scales."""
import hashlib
import numpy as np
import residual_features

SCALES = (1., 1.5, 2., 2.5, 3., 4.)
A0, RHO = 2., 3.
TAUS = tuple(j / 100 for j in range(-100, 201))
METHODS = ('Full', 'G', 'L', 'D', 'G_plus_L', 'VideoMeanB', 'RefResidualUnclipped')


def dense_support(peaks, values, n, delta):
    """Distance first, higher support second, original index on complete ties."""
    support = np.zeros(n)
    distance = np.full(n, delta + 1, dtype=int)
    for p, value in zip(peaks, values):
        times = np.arange(max(0, p-delta), min(n, p+delta+1))
        dist = np.abs(times-p)
        update = (dist < distance[times]) | ((dist == distance[times]) & (value > support[times]))
        times, dist = times[update], dist[update]
        support[times], distance[times] = value, dist
    return support


def compute(base, reference=A0, radius=RHO, chunk_size=4096):
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    peaks, gl = base.evidence(reference, radius)
    widths = list(base.effective_scales)
    ref = next(i for i, width in enumerate(widths)
               if base.effective_scales[width] is base.scale_data[reference])
    n, k = len(base.response), base.k
    w, delta = max(1, round(radius*k)), max(1, round(.5*k))
    maps = np.asarray([dense_support(base.effective_scales[width][0],
                      base.local_cache[width, w], n, delta) for width in widths])
    np.testing.assert_allclose(np.median(maps[:, peaks], axis=0), gl[:, 1], atol=1e-12, rtol=0)
    l0 = maps[ref, peaks]
    q = w + delta + int(np.ceil(max(widths)/2))
    times = np.arange(q, max(q, n-q))
    others = np.delete(maps, ref, axis=0)
    background = gl[:, 1].copy()
    counts = np.zeros(len(peaks), dtype=int)
    reasons = []
    # One candidate and a bounded time block at once, never N x T x A.
    for i, p in enumerate(peaks):
        controls = times[np.abs(times-p) > 2*q]
        counts[i] = len(controls)
        reason = 'single_physical_scale' if len(widths) == 1 else 'no_controls' if not len(controls) else ''
        reasons.append(reason)
        if reason:
            continue
        total = 0.
        for start in range(0, len(controls), chunk_size):
            t = controls[start:start+chunk_size]
            votes = np.vstack((np.full(len(t), l0[i]), others[:, t]))
            total += np.median(votes, axis=0).sum(dtype=np.float64)
        background[i] = total / len(controls)
    d = gl[:, 1] - background
    full = gl[:, 0] + gl[:, 1] - background
    abstain = np.asarray([bool(r) for r in reasons], dtype=bool)
    # Exact restoration of G, including floating-point representation.
    full[abstain] = gl[abstain, 0]
    for values in (gl, l0, background):
        if not np.isfinite(values).all() or np.any(values < -1e-12) or np.any(values > 1+1e-12):
            raise RuntimeError('Nonfinite or out-of-range evidence')
    return dict(peaks=peaks.tolist(), G=gl[:, 0].tolist(), L=gl[:, 1].tolist(),
                L0=l0.tolist(), B=background.tolist(), D=d.tolist(), S=full.tolist(),
                control_count=counts.tolist(), abstention=reasons, physical_widths=widths,
                reference_width=widths[ref], window=w, tolerance=delta, protection_radius=q,
                a0=reference, rho=radius, n=n,
                video_mean_B=float(background.mean()) if len(background) else 0.)


def score(data, method):
    g, l, l0, b = (np.asarray(data[k], float) for k in ('G', 'L', 'L0', 'B'))
    if method == 'Full':
        return np.asarray(data['S'], float)
    if method == 'G':
        return g
    if method == 'L':
        return l
    if method == 'D':
        return l-b
    if method == 'G_plus_L':
        return g+l
    if method == 'VideoMeanB':
        return g+l-data['video_mean_B']
    if method == 'RefResidualUnclipped':
        return g+l-l0
    raise ValueError(method)


class FeatureView(residual_features.FeatureView):
    def shift_evidence(self, reference=A0, radius=RHO):
        key = ('shift', reference, radius)
        if key not in self.details:
            self.details[key] = compute(self.base, reference, radius)
        return self.details[key]

    def selected_peaks(self, config):
        data = self.shift_evidence(config.reference_scale, config.local_radius)
        values = score(data, config.method)
        keep = values >= config.threshold
        self.owner.calls += 1
        if self.owner.capture:
            self.owner.trace.append(dict(response_sha256=self.key[2], k=self.key[0],
                **data, score=values.tolist(), retained=keep.tolist()))
        return np.asarray(data['peaks'], int)[keep]


class FusionCore(residual_features.FusionCore):
    def GLSDFeatures(self, response, k):
        array = np.ascontiguousarray(response, dtype=float)
        key = (int(k), array.shape, hashlib.sha256(array.tobytes()).hexdigest())
        if key not in self.cache:
            self.cache[key] = FeatureView(self.base.GLSDFeatures(array, k), key, self)
        return self.cache[key]
