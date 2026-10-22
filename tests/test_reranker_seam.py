"""Unit tests for the re-ranking stage: it may improve the order, never cost the answer.

Run from the repository root:  python -m unittest discover -s tests
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import oncall_kb  # noqa: E402
import oncall_reranker  # noqa: E402

DOC = """Faded or Missing Print
If the print is faded, patchy or missing lines, the nozzles are blocked. Wipe the nozzle plate
with a lint-free cloth and run a cleaning cycle.

Battery Light
A flashing battery light means the battery is low. Charge the unit for two hours."""


def seed(db: Path) -> None:
    oncall_kb.ingest_text(db, device_family="demo", doc_title="Demo manual", text=DOC)


class RerankerSeamTests(unittest.TestCase):
    ENV = ("ONCALL_RERANKER", "ONCALL_RERANKER_POOL", "ONCALL_RERANKER_ALPHA",
           "ONCALL_RERANKER_MIN_SCORE")

    def setUp(self):
        self._saved = {key: os.environ.pop(key, None) for key in self.ENV}
        oncall_reranker.reset()

    def tearDown(self):
        oncall_reranker.reset()
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    @staticmethod
    def _pool():
        return [{"body": "battery light", "score": 9.0},
                {"body": "clean the nozzles", "score": 5.0},
                {"body": "print density", "score": 1.0}]

    def test_closed_seam_keeps_first_stage_order_and_leaves_inputs_untouched(self):
        pool = self._pool()
        results, mode = oncall_reranker.rerank("q", pool, limit=2)
        self.assertEqual(mode, "bm25")
        self.assertEqual([hit["body"] for hit in results], ["battery light", "clean the nozzles"])
        self.assertNotIn("bm25_rank", pool[0])           # the caller's dicts are not mutated

    def test_scorer_reorders_on_raw_scores_not_rounded_display_values(self):
        oncall_reranker.set_scorer(lambda q, p: [0.12341, 0.12344, 0.0], name="t")
        results, mode = oncall_reranker.rerank("q", self._pool(), limit=3)
        self.assertEqual(mode, "t")
        self.assertEqual(results[0]["body"], "clean the nozzles")
        self.assertEqual(results[0]["bm25_rank"], 2)

    def test_failing_scorer_degrades_to_first_stage_order(self):
        def boom(query, passages):
            raise RuntimeError("model unavailable")
        oncall_reranker.set_scorer(boom, name="t")
        results, mode = oncall_reranker.rerank("q", self._pool(), limit=3)
        self.assertEqual(mode, "bm25_fallback:RuntimeError")
        self.assertEqual(results[0]["body"], "battery light")

    def test_alpha_one_reproduces_first_stage_and_alpha_blends(self):
        oncall_reranker.set_scorer(lambda q, p: [0.0, 1.0, 0.9], name="t")
        only_bm25, _ = oncall_reranker.rerank("q", self._pool(), limit=3, alpha=1.0)
        self.assertEqual([h["body"] for h in only_bm25],
                         ["battery light", "clean the nozzles", "print density"])
        only_model, _ = oncall_reranker.rerank("q", self._pool(), limit=3, alpha=0.0)
        self.assertEqual(only_model[0]["body"], "clean the nozzles")
        blended, _ = oncall_reranker.rerank("q", self._pool(), limit=3, alpha=0.5)
        # Normalised: bm25 (1.0, .5, 0) and model (0, 1, .9) -> fused (.5, .75, .45).
        self.assertEqual([h["body"] for h in blended],
                         ["clean the nozzles", "battery light", "print density"])

    def test_low_confidence_abstains_and_search_reports_an_honest_miss(self):
        oncall_reranker.set_scorer(lambda q, p: [-3.0] * len(p), name="t")
        results, mode = oncall_reranker.rerank("q", self._pool(), limit=3, min_score=0.0)
        self.assertEqual(results, [])
        self.assertEqual(mode, "t:abstained")
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "kb.db"
            seed(db)
            os.environ["ONCALL_RERANKER_MIN_SCORE"] = "0"
            out = oncall_kb.kb_search(db, "faded patchy print nozzles", device_family="demo")
            self.assertFalse(out["found"])
            self.assertEqual(out["rerank"], "t:abstained")

    def test_pool_size_comes_from_the_environment_and_latency_is_reported(self):
        os.environ["ONCALL_RERANKER_POOL"] = "40"
        oncall_reranker.set_scorer(lambda q, p: [1.0] * len(p), name="t")
        self.assertEqual(oncall_reranker.candidate_pool(3), 40)
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "kb.db"
            seed(db)
            out = oncall_kb.kb_search(db, "faded patchy print nozzles", device_family="demo")
            self.assertTrue(out["found"])
            self.assertGreaterEqual(out["rerank_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
