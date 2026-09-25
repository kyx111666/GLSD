"""Read intermediate values from the sealed GLSDFeatures instance."""
import hashlib
import numpy as np
import run_p8_phase2 as phase

METHODS = ('G', 'L', 'Mean', 'RefResidual', 'PositiveResidual')
LOCKED = ('no_local', 'no_G', 'no_negative')


def score(g, l, l0, method):
    if method in ('G', 'no_local'):
        return g
    if method == 'L':
        return l
    if method == 'Mean':
        return (g + l) / 2
    if method.startswith('AsymResidual_l'):
        token = method.split('_l', 1)[1]
        try:
            negative_weight = float(token.replace('p', '.'))
        except ValueError as exc:
            raise ValueError('invalid asymmetric residual method: '+method) from exc
        delta = np.maximum(l - l0, 0.) - negative_weight * np.maximum(l0 - l, 0.)
        return np.clip(g + delta, 0, 1)
    delta = l - l0
    if method in ('PositiveResidual', 'no_negative'):
        delta = np.maximum(delta, 0)
    if method == 'no_G':
        return np.clip(delta, 0, 1)
    if method not in ('RefResidual', 'PositiveResidual', 'no_negative'):
        raise ValueError(method)
    return np.clip(g + delta, 0, 1)


class FeatureView(phase.FeatureView):
    def __init__(self, base, key, owner):
        super().__init__(base, key, owner)
        self.details = {}

    def residual_evidence(self, reference, radius):
        key = (reference, radius)
        if key in self.details:
            return self.details[key]
        peaks, evidence = self.evidence(reference, radius)
        # The sealed evidence() call fills local_cache. Reuse those actual values.
        window = max(1, round(radius * self.key[0]))
        tolerance = max(1, round(.5 * self.key[0]))
        physical = self.base.effective_scales
        ref_data = self.base.scale_data[reference]
        widths = list(physical)
        reference_width = next((w for w, data in physical.items() if data is ref_data), None)
        if reference_width is None:
            raise RuntimeError('sealed core reference/physical-scale identity differs; inspect core, do not guess L0')
        if not np.array_equal(peaks, ref_data[0]) or not np.array_equal(evidence[:, 0], ref_data[3]):
            raise RuntimeError('sealed reference candidates/G mismatch')
        per_scale, matching, spreads = [], [], []
        for width, (other_peaks, smooth, spread, _) in physical.items():
            values = np.asarray(self.base.local_cache[width, window])
            # Confirm the cached intermediate really is the documented local contrast.
            for peak, value in zip(other_peaks, values):
                left = np.min(smooth[max(0, peak-window):peak+1])
                right = np.min(smooth[peak:min(len(smooth), peak+window+1)])
                expected = max(0., float(smooth[peak])-max(float(left), float(right))) / spread
                if not np.isclose(value, expected, rtol=0, atol=1e-12):
                    raise RuntimeError('sealed local-support definition differs')
            aligned, matched = [], []
            for peak in peaks:
                available = np.flatnonzero(np.abs(other_peaks-peak) <= tolerance)
                if not len(available):
                    aligned.append(0.)
                    matched.append(-1)
                else:
                    chosen = min(available, key=lambda i: (abs(int(other_peaks[i])-int(peak)), -values[i]))
                    aligned.append(float(values[chosen]))
                    matched.append(int(other_peaks[chosen]))
            per_scale.append(aligned)
            matching.append(matched)
            spreads.append(float(spread))
        local = np.asarray(per_scale, float).reshape(len(widths), len(peaks)).T
        matched = np.asarray(matching, int).reshape(len(widths), len(peaks)).T
        if not np.array_equal(np.median(local, axis=1), evidence[:, 1]):
            raise RuntimeError('reconstructed median differs from sealed L; alignment/core mismatch')
        l0 = local[:, widths.index(reference_width)]
        if not np.array_equal(l0, self.base.local_cache[reference_width, window]):
            raise RuntimeError('reference local support did not match its own peak')
        values = dict(peaks=peaks.tolist(), G=evidence[:, 0].tolist(), L=evidence[:, 1].tolist(),
                      L0=l0.tolist(), D=(evidence[:, 1]-l0).tolist(), physical_widths=widths,
                      reference_width=reference_width, window=window, tolerance=tolerance,
                      per_scale_support=local.tolist(), matched_peaks=matched.tolist(),
                      missing_support=(matched < 0).tolist(),
                      matched_distance=np.where(matched < 0, -1, np.abs(matched-peaks[:, None])).tolist(),
                      per_scale_local_contrast=(local*np.asarray(spreads)).tolist(),
                      per_scale_spread=spreads)
        self.details[key] = values
        return values

    def selected_peaks(self, config):
        data = self.residual_evidence(config.reference_scale, config.local_radius)
        g, l, l0 = (np.asarray(data[k], float) for k in ('G', 'L', 'L0'))
        values = score(g, l, l0, config.method)
        keep = values >= config.threshold
        self.owner.calls += 1
        if self.owner.capture:
            self.owner.trace.append(dict(response_sha256=self.key[2], k=self.key[0],
                **data, score=values.tolist(), retained=keep.tolist()))
        return np.asarray(data['peaks'], int)[keep]


class FusionCore(phase.FusionCore):
    def GLSDFeatures(self, response, k):
        array = np.ascontiguousarray(np.asarray(response, dtype=float))
        key = (int(k), array.shape, hashlib.sha256(array.tobytes()).hexdigest())
        if key not in self.cache:
            self.cache[key] = FeatureView(self.base.GLSDFeatures(response, k), key, self)
        return self.cache[key]
