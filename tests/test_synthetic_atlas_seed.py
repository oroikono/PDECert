import json
import tempfile
import unittest
from pathlib import Path

from experiments.build_synthetic_atlas_seed import build_bundles, write_bundles
from pdecert import (
    Status,
    case_from_dict,
    load_atlas,
    load_record_bundle,
    verify,
)


class SyntheticAtlasSeedTests(unittest.TestCase):
    def test_generated_bundles_validate_and_remain_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = root / "records"
            write_bundles(records)
            (root / "atlas.json").write_text(
                json.dumps(
                    {
                        "atlas_version": 1,
                        "description": "test atlas",
                        "name": "test atlas",
                    }
                )
            )
            atlas = load_atlas(root)

        self.assertEqual(len(atlas["records"]), 6)
        self.assertTrue(
            all(record["annotation"]["status"] == "pending" for record in atlas["records"])
        )
        self.assertTrue(all(record["annotation"]["verdict"] is None for record in atlas["records"]))

    def test_machine_outcomes_are_kept_separate_from_pending_labels(self):
        expected = {
            "synthetic-heat-exact-control": Status.PROVED,
            "synthetic-heat-pde-only-boundary": Status.REFUTED,
            "synthetic-heat-fixed-grid-alias": Status.REFUTED,
            "synthetic-heat-hidden-singularity": Status.REFUTED,
            "synthetic-heat-single-parameter": Status.REFUTED,
            "synthetic-heat-below-tolerance": Status.INCONCLUSIVE,
        }
        for record_id, status in expected.items():
            record = load_record_bundle(Path("corpus/community/records") / record_id)
            case = case_from_dict(record["case"])
            self.assertEqual(
                verify(case.problem, case.candidate_fields).status,
                status,
                record_id,
            )

    def test_checked_in_seed_matches_the_generator(self):
        records = Path("corpus/community/records")
        generated = build_bundles()

        checked_in = {path.name for path in records.iterdir() if path.is_dir()}
        self.assertTrue(set(generated).issubset(checked_in))
        for record_id, expected in generated.items():
            actual = load_record_bundle(records / record_id)
            self.assertEqual(actual["id"], record_id)
            self.assertEqual(actual["case"], expected.case)
            self.assertEqual(actual["origin"], expected.metadata["origin"])
            self.assertEqual(actual["annotation"], expected.metadata["annotation"])
            self.assertEqual(actual["raw_output"], expected.raw_output)
            self.assertEqual(
                actual["output_sha256"],
                expected.metadata["output_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
