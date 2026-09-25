"""Expanded references must not silently change the sealed L support votes."""
import importlib.util
import contextlib
import csv
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np

import run_full_fusion_tuning as full
import run_one_to_one_full_tuning as runner
import structure_expansion as expanded


class StructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[2] / "official_BoostingVRME/boosting_official_glds_full.py"
        spec = importlib.util.spec_from_file_location("structure_test_sealed_core", path)
        cls.base = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.base
        spec.loader.exec_module(cls.base)

    def setUp(self):
        self.previous = full.SCALES, full.RADII, full.THRESHOLDS

    def tearDown(self):
        full.SCALES, full.RADII, full.THRESHOLDS = self.previous

    def test_existing_evidence_exact_and_support_votes_unchanged(self):
        rng = np.random.default_rng(37)
        for n, k in ((3, 1), (13, 8), (181, 5), (181, 17)):
            for response in (np.ones(n), rng.uniform(size=n)):
                old = self.base.GLSDFeatures(response, k)
                new = expanded.ExpandedCore(self.base, expanded.EXPANDED_SCALES).GLSDFeatures(response, k)
                self.assertEqual(list(new.effective_scales), list(old.effective_scales))
                for a0 in full.LEGACY_SCALES:
                    for rho in expanded.EXPANDED_RADII:
                        for got, expected in zip(new.evidence(a0, rho), old.evidence(a0, rho)):
                            np.testing.assert_array_equal(got, expected)
                for a0 in expanded.EXPANDED_SCALES:
                    for rho in expanded.EXPANDED_RADII:
                        peaks, evidence = new.evidence(a0, rho)
                        self.assertEqual(evidence.shape, (len(peaks), 2))
                        self.assertTrue(np.isfinite(evidence).all())
                        self.assertTrue(((evidence >= 0) & (evidence <= 1 + 1e-12)).all())

    def test_new_reference_local_evidence_uses_only_original_support(self):
        response = np.random.default_rng(12).uniform(size=201)
        k, a0, rho = 5, 3., 1.5
        old = self.base.GLSDFeatures(response, k)
        new = expanded.ExpandedCore(self.base, expanded.EXPANDED_SCALES).GLSDFeatures(response, k)
        peaks, values = new.evidence(a0, rho)
        # Independent direct calculation of aligned local support at new peaks.
        expected = []
        for peak in peaks:
            support = []
            for other, curve, spread, _ in old.effective_scales.values():
                possible = []
                for q in other:
                    if abs(int(q)-int(peak)) > max(1, round(.5*k)):
                        continue
                    w = max(1, round(rho*k))
                    drop = max(0., float(curve[q])-max(float(min(curve[max(0,q-w):q+1])),
                                                     float(min(curve[q:min(len(curve),q+w+1)])))) / spread
                    possible.append((abs(int(q)-int(peak)), -drop))
                support.append(-min(possible)[1] if possible else 0.)
            expected.append(np.median(support))
        np.testing.assert_array_equal(values[:, 1], expected)

    def test_integer_aliases_share_evidence_and_short_curves_are_capped(self):
        new = expanded.ExpandedCore(self.base, expanded.EXPANDED_SCALES).GLSDFeatures([0., 1., 0.], 1)
        self.assertIs(new.evidence(.5, .5), new.evidence(1., 1.))
        self.assertIs(new.evidence(3., 2.), new.evidence(4., 2.))

    def test_grid_preserves_old_order_and_subsets(self):
        runner.configure_threshold_grid("refined")
        runner.configure_structure_grid("legacy")
        original = full.grids()
        runner.configure_structure_grid("expanded")
        grids = full.grids()
        self.assertEqual({m: len(g) for m, g in grids.items()},
            dict(G=133, L=798, EqualMean=798, WeightedMean=4788, DominantEvidence=4788))
        for method, grid in grids.items():
            self.assertEqual(grid[:len(original[method])], original[method])
            self.assertEqual(expanded.subset_indexes(grid, "original"), list(range(len(original[method]))))
            self.assertTrue(all(grid[i].local_radius in full.LEGACY_RADII
                                for i in expanded.subset_indexes(grid, "a0_only")))
            self.assertTrue(all(grid[i].reference_scale in full.LEGACY_SCALES
                                for i in expanded.subset_indexes(grid, "rho_only")))
        full.verify_grid(SimpleNamespace(locked_grid=lambda base: base.configuration_grid()), self.base)

    def test_subset_selection_excludes_subject_and_writes_complete_outputs(self):
        grid = [full.Config("L", 0, 1., 1., .5), full.Config("L", 1, 3., 1., .5),
                full.Config("L", 2, 1., 4., .5)]
        table = np.array([[[1,0,1],[1,0,1]], [[2,0,0],[0,3,2]], [[0,3,2],[2,0,0]]])
        def decode(context, core, si, subject, config, recognition, setting, method):
            counts = tuple(table[config.config_id, si])
            return counts, counts, [], []
        with TemporaryDirectory() as directory:
            audit = expanded.StructureAudit("synthetic", Path(directory), ["A", "B"])
            with patch.object(full, "decode_with_context", side_effect=decode):
                audit.evaluate("L", grid, table, None, None)
            selected = {(r["variant"], r["subject"]):r["config_id"] for r in audit.selections}
            self.assertEqual(selected, {("original","A"):0, ("original","B"):0,
                ("a0_only","A"):0, ("a0_only","B"):1,
                ("rho_only","A"):2, ("rho_only","B"):0})
            self.assertEqual(len(audit.rows), 12)
            self.assertTrue((Path(directory)/"structure_subset_predictions.jsonl.gz").is_file())

    def test_previous_run_hash_mismatch_stops_before_decoding(self):
        with TemporaryDirectory() as directory:
            audit = expanded.StructureAudit("synthetic", Path(directory), ["A", "B"])
            audit.reference = {"methods":{"L":{"raw_counts_sha256":"wrong"}}}
            with self.assertRaisesRegex(RuntimeError, "Previous refined search counts"):
                audit.evaluate("L", [full.Config("L",0,1.,1.,.5)], np.zeros((1,2,3)), None, None)

    def test_expanded_runner_completes_mean_replay_and_all_subset_artifacts(self):
        runner.configure_structure_grid("expanded")
        full.THRESHOLDS = (.05, .5, .95)
        responses = [np.random.default_rng(seed).uniform(size=71) for seed in (2, 3)]
        def decode(records, si, config, core, metric, official, recognition):
            peaks = core.GLSDFeatures(records[si], 5).selected_peaks(config)
            counts = (min(2, len(peaks)), max(0, len(peaks)-2), max(0, 2-len(peaks)))
            return counts, [peaks.tolist()], counts, [], None
        fake_runner = SimpleNamespace(decode_glsd_subject=decode,
            full_counts_from_official_synergy=lambda raw, pred, gt: raw)
        context = (fake_runner, self.base, None, None, responses, None, ["A", "B"], None)
        def replay(kind, context, core, selected, subjects):
            result = {}
            for si, subject in enumerate(subjects):
                raw = decode(responses, si, selected[subject], core, None, None, True)[0]
                result[subject] = raw, raw
            return result
        ora = SimpleNamespace(SPECS={"synthetic":dict(videos=2, gt=4,
                source=Path(self.base.__file__), core=Path(self.base.__file__))},
            locked_grid=lambda base:base.configuration_grid(),
            context_video_and_gt_counts=lambda kind, context:(2,4), sha256=lambda path:"synthetic",
            locked_glsd_subject_counts=replay)
        grids = {}
        for method in full.METHODS:
            grids[method] = [full.Config(method, i, a0, rho, .5,
                alpha=.4 if method == "WeightedMean" else None,
                beta=.8 if method == "DominantEvidence" else None)
                for i, (a0, rho) in enumerate(((1.,1.),(3.,1.),(1.,4.),(3.,4.)))]
        with TemporaryDirectory() as directory:
            output = Path(directory)/"synthetic"
            with patch.object(full, "grids", return_value=grids), contextlib.redirect_stdout(io.StringIO()):
                result, paired = full.run_setting(ora, "synthetic", output, prepared_context=context,
                    evaluation_protocol="synthetic_one_to_one", expanded_structure=True)
            self.assertEqual(len(result),5)
            self.assertEqual(len(paired),4)
            self.assertTrue(json.loads((output/"completion.json").read_text())["completed"])
            with (output/"structure_subset_summary.csv").open() as handle:
                rows=list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4*5*2)
            self.assertEqual({r["variant"] for r in rows}, {"original","a0_only","rho_only","expanded"})
            with (output/"structure_feature_diagnostics.csv").open() as handle:
                diagnostics=list(csv.DictReader(handle))
            self.assertEqual(len(diagnostics), 2*7*6)
            self.assertTrue(all(r["support_widths"] == "5|8|10" for r in diagnostics))


if __name__ == "__main__":
    unittest.main()
