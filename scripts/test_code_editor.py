"""
NOT part of the agent pipeline. Verifies the safety mechanism the whole
orchestrator loop depends on: a broken patch must be rejected and rolled
back automatically, and a good patch must be applied and kept. This is
tested with hand-written patches here (not real LLM output) specifically
to isolate "does the rollback mechanism work" from "does Claude write good
code" — two different questions.
"""
from __future__ import annotations

from agent.code_editor import EDITABLE_FILES, apply_and_smoke_test

FEATURES_PATH = EDITABLE_FILES["features"]

BROKEN_PATCH = """
def build_features(df, video_features=None, allow_leaky_columns=None):
    this is not valid python at all !!! ][
    return df
"""

# Valid Python, valid signature, but let's make sure it functionally
# still can produce a working pipeline: keep the real body, just add a
# harmless comment so the diff is non-trivial and it still passes.
def _make_working_patch(original: str) -> str:
    return "# agent test patch\n" + original


def main() -> None:
    original_content = FEATURES_PATH.read_text(encoding="utf-8")

    print("[code_editor] Applying a deliberately BROKEN patch...")
    broken_result = apply_and_smoke_test("features", BROKEN_PATCH, smoke_test_module="pipeline.smoke_test", timeout_s=60)
    print(f"  applied={broken_result.applied} rolled_back={broken_result.rolled_back} error={broken_result.error!r}")
    assert broken_result.applied is False
    assert broken_result.rolled_back is True

    restored = FEATURES_PATH.read_text(encoding="utf-8")
    assert restored == original_content, "FILE WAS NOT PROPERLY ROLLED BACK — this is a critical safety bug"
    print("[code_editor] Rollback verified: file content matches pre-patch original exactly.")

    print("\n[code_editor] Applying a VALID patch...")
    good_patch = _make_working_patch(original_content)
    good_result = apply_and_smoke_test("features", good_patch, smoke_test_module="pipeline.smoke_test", timeout_s=60)
    print(f"  applied={good_result.applied} smoke_test_passed={good_result.smoke_test_passed}")
    assert good_result.applied is True
    assert "SMOKE_TEST_METRICS" in good_result.smoke_test_output

    kept = FEATURES_PATH.read_text(encoding="utf-8")
    assert kept == good_patch, "valid patch was not actually kept on disk"
    print("[code_editor] Valid patch verified: kept on disk, smoke test passed with real metrics.")

    # restore the original file so this test doesn't leave the repo modified
    FEATURES_PATH.write_text(original_content, encoding="utf-8")
    print("\n[code_editor] Restored original features.py. ALL CODE EDITOR TESTS PASSED")


if __name__ == "__main__":
    main()
