"""Reliable local union: read sealed physical-scale caches; never edit the core."""
from dataclasses import dataclass, asdict
from fractions import Fraction
import hashlib
import numpy as np
import run_p8_threshold_control as prior

phase, shared = prior.phase, prior.shared
PROTOCOL = 'reliable_local_union_v1'
BETAS = (.5, 1., 2.)
# T first preserves the old L tie preference; additional exact rational values follow.
T = tuple(Fraction(i, 20) for i in range(1, 20))
U = T + tuple(sorted((set(T) | {t/2 for t in T} | {2*t for t in T}) - set(T)))
LOCKED = ('Full_no_L', 'Full_no_G', 'Full_no_R')

@dataclass(frozen=True)
class Config(phase.Config):
    beta: float = 1.
    p: str = 'inf'  # used ONLY by inherited rescaled-G controls


def grids():
    result = {}
    for method in ('Full', 'G', 'L', 'L_rel', 'NoR', 'Mean', 'L_old'):
        radii = (1.,) if method == 'G' else phase.RADII
        betas = BETAS if method in ('Full', 'NoR') else (1.,)
        taus = U if method in ('L', 'L_rel') else T
        result[method] = [Config(method, i, a, rho, float(t), beta) for i, (a,rho,beta,t) in enumerate(
            (a,rho,beta,t) for a in phase.SCALES for rho in radii for beta in betas for t in taus)]
    return result


def reliability(distances, reference_index, k):
    d = np.asarray(distances, dtype=float)
    if d.ndim != 2 or not 0 <= reference_index < d.shape[1] or k <= 0:
        raise ValueError('invalid scale distances/reference/k')
    other = np.delete(d, reference_index, axis=1)
    result = np.zeros(len(d), dtype=float)
    if other.shape[1]:
        complete = np.all(other >= 0, axis=1)
        result[complete] = np.exp(-np.mean(other[complete] / k, axis=1))
    return result


def score_keep(values, r, c):
    phase.score(values, 'G')
    r = np.asarray(r, dtype=float)
    if r.shape != (len(values),) or not np.isfinite(r).all() or np.any((r < 0) | (r > 1)):
        raise ValueError('invalid reliability')
    if c.beta not in BETAS or not np.isfinite(c.threshold) or c.threshold <= 0:
        raise ValueError('outside frozen beta/positive threshold domain')
    g, l = np.asarray(values).T
    local = r*l
    if c.method in ('G', 'G_union', 'Full_no_L'): scores = g
    elif c.method in ('L', 'L_old'): scores = l
    elif c.method == 'L_rel': scores = local
    elif c.method == 'Mean': scores = (g+l)/2
    elif c.method == 'Full_no_G': scores = c.beta*local
    elif c.method == 'Full': scores = np.maximum(g, c.beta*local)
    elif c.method in ('NoR', 'Full_no_R'): scores = np.maximum(g, c.beta*l)
    elif c.method.startswith('G_rescaled_p'):
        import run_joint_gm as joint
        factor = joint.factor(c.p)
        return g/factor, g >= factor*c.threshold
    else: raise ValueError(c.method)
    return scores, scores >= c.threshold


class FeatureView(phase.FeatureView):
    def __init__(self, base, key, owner):
        super().__init__(base, key, owner)
        self.reliable_cache = {}

    def reliable_evidence(self, scale, radius):
        pair = scale, radius
        if pair in self.reliable_cache: return self.reliable_cache[pair]
        peaks, values = self.evidence(scale, radius)  # unchanged sealed implementation + mean replay
        widths = list(self.base.effective_scales)
        ref = [i for i,w in enumerate(widths)
               if self.base.effective_scales[w] is self.base.scale_data[scale]]
        if len(ref) != 1: raise RuntimeError('reference physical-scale identity is ambiguous')
        window = max(1, round(radius*self.k))
        tolerance = max(1, round(.5*self.k))
        matched = np.full((len(peaks), len(widths)), -1, dtype=int)
        distances = np.full_like(matched, -1)
        local = np.zeros(matched.shape, dtype=float)
        for j,w in enumerate(widths):
            other = self.base.effective_scales[w][0]
            support = self.base.local_cache[w,window]
            for i,p in enumerate(peaks):
                options = np.flatnonzero(np.abs(other-p) <= tolerance)
                if len(options):
                    chosen = min(options, key=lambda q:(abs(int(other[q])-int(p)), -support[q]))
                    matched[i,j] = other[chosen]
                    distances[i,j] = abs(int(other[chosen])-int(p))
                    local[i,j] = support[chosen]
        if not np.array_equal(np.median(local,axis=1), values[:,1]):
            raise RuntimeError('per-scale matching does not exactly reproduce sealed L')
        if not np.array_equal(matched[:,ref[0]], peaks):
            raise RuntimeError('reference peaks did not match themselves')
        r = reliability(distances, ref[0], self.k)
        data = dict(peaks=peaks, values=values, R=r, L_rel=r*values[:,1], widths=widths,
                    reference_width=widths[ref[0]], reference_index=ref[0], tolerance=tolerance,
                    matched_peaks=matched, distances=distances, per_scale_local=local,
                    B_empty=len(widths)==1)
        self.reliable_cache[pair] = data
        return data

    def selected_peaks(self,c):
        d = self.reliable_evidence(c.reference_scale,c.local_radius)
        peaks, values, r = d['peaks'], d['values'], d['R']
        scores, keep = score_keep(values,r,c)
        self.owner.calls += 1
        if self.owner.capture:
            ga = values[:,0] >= c.threshold
            la = d['L_rel'] >= c.threshold/c.beta
            # These branch flags always describe the Full decision at these parameters.
            self.owner.trace.append(dict(response_sha256=self.key[2], k=self.k,
                peaks=peaks.tolist(), G=values[:,0].tolist(), L=values[:,1].tolist(), R=r.tolist(),
                L_rel=d['L_rel'].tolist(), score=scores.tolist(), retained=keep.tolist(),
                G_accept=ga.tolist(), reliable_L_accept=la.tolist(), both_accept=(ga&la).tolist(),
                full_accept_source=np.where(ga&la,'both',np.where(ga,'G',np.where(la,'reliable_L','neither'))).tolist(),
                branch_flags_semantics='Full branches at recorded parameters; retained follows actual method',
                effective_widths=d['widths'], reference_width=d['reference_width'],
                tolerance=d['tolerance'], B_empty=d['B_empty'],
                matched_peaks=d['matched_peaks'].tolist(), distances=d['distances'].tolist(),
                per_scale_local=d['per_scale_local'].tolist()))
        return peaks[keep]


class FusionCore(prior.FusionCore):
    def GLSDFeatures(self,response,k):
        array = np.ascontiguousarray(np.asarray(response,dtype=float))
        key = int(k), array.shape, hashlib.sha256(array.tobytes()).hexdigest()
        if key not in self.cache:
            self.cache[key] = FeatureView(self.base.GLSDFeatures(response,k),key,self)
        return self.cache[key]
