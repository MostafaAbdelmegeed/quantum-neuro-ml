import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("external_inventory", Path("scripts/external_metadata_inventory.py"))
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


def test_canonical_ids_and_header_normalization():
    assert inventory.canonical_id("0051209") == inventory.canonical_id("sub-51209")
    fields, rows = inventory.parse_table(b"participant_id\tage_at_scan \nsub-1\t12\n")
    assert fields == ["participant_id", "age_at_scan"] and rows[0]["age_at_scan"] == "12"
    with pytest.raises(ValueError, match="Invalid canonical"):
        inventory.canonical_id("sub-x51209")
    with pytest.raises(ValueError, match="duplicate"):
        inventory.parse_table(b"age\tage \n12\t12\n")
    _, indexed = inventory.parse_table(b",SUB_ID\n0,0051209\n1,0051210\n", ",", allow_row_index=True)
    assert indexed[0]["sub_id"] == "0051209"
    with pytest.raises(ValueError, match="sequential"):
        inventory.parse_table(b",SUB_ID\n9,0051209\n", ",", allow_row_index=True)
    assert inventory.parse_table(b"age_at_scan\n12\xa0\n")[1][0]["age_at_scan"] == "12"
    assert inventory.decode_metadata(b"age\n12\xa0\n")[1] == "latin1_nbsp_only"
    with pytest.raises(UnicodeDecodeError):
        inventory.decode_metadata(b"age\n12\x81\n")


def test_overlap_longitudinal_duplicates_and_missing_values_are_excluded():
    def row(identity, *, site="NYU_1", age="12", sex="1", ndar=""):
        return {"participant_id": identity, "site_id": site, "age_at_scan": age,
                "sex": sex, "dx_group": "1", "ndar_guid": ndar}
    rows = [row("sub-51209"), row("sub-60001"), row("sub-60002", age="NaN"),
            row("sub-60003", sex="-9999"), row("sub-60004", ndar="NDAR_DUP"),
            row("sub-60005", ndar="NDAR_DUP"), row("sub-60006"), row("sub-60006")]
    collections = {"ABIDEII-NYU_1": ([], rows),
                   "ABIDEII-UCLA_Long": ([], [row("sub-60100", site="UCLA_Long")])}
    records = inventory.candidate_rows(collections, [{"sub_id": "0051209"}], {"ABIDEII-UCLA_Long"})
    candidates = [r for r in records if r["metadata_candidate"]]
    assert [r["canonical_id"] for r in candidates] == [60001]
    assert any("abide1_numeric_identity_overlap" in r["reasons"] for r in records)
    assert any("known_longitudinal_collection" in r["reasons"] for r in records)
    assert any("ambiguous_repeated_ndar_guid" in r["reasons"] for r in records)


def test_cached_metadata_checksums_and_missing_offline_inputs(tmp_path):
    import hashlib
    import json

    path = tmp_path / "participants.tsv"
    path.write_bytes(b"metadata")
    record = {"url": "https://example.org/participants.tsv", "status": 200,
              "sha256": hashlib.sha256(b"metadata").hexdigest()}
    path.with_suffix(".tsv.receipt.json").write_text(json.dumps(record))
    assert inventory.cached_fetch(record["url"], path, True)["status"] == 200
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        inventory.cached_fetch(record["url"], path, True)
    with pytest.raises(ValueError, match="incomplete"):
        inventory.cached_fetch(record["url"], tmp_path / "missing.tsv", True)


def test_captured_inputs_must_match_pre_retrieval_identity_and_source_receipts():
    import hashlib
    import json
    from types import SimpleNamespace

    configuration, script = b"configuration", b"script"
    source = {"status": 200, "sha256": "source", "url": "https://example.org/index"}
    captured = {"development_metadata": {"sha256": "development"},
                "configuration": {"sha256": hashlib.sha256(configuration).hexdigest()},
                "runner_script": {"sha256": hashlib.sha256(script).hexdigest()},
                "raw_index": {"sha256": "source"}}
    run = SimpleNamespace(record={"inputs": captured}, input_payloads={"raw_index_receipt": json.dumps(source).encode()})
    config = {"abide1_metadata_sha256": "development"}
    inventory.validate_captured_inputs(run, config, configuration, script, {"raw_index": source})
    captured["development_metadata"]["sha256"] = "changed"
    with pytest.raises(ValueError, match="frozen identity"):
        inventory.validate_captured_inputs(run, config, configuration, script, {"raw_index": source})
    captured["development_metadata"]["sha256"] = "development"
    captured["raw_index"]["sha256"] = "changed"
    with pytest.raises(ValueError, match="source differs"):
        inventory.validate_captured_inputs(run, config, configuration, script, {"raw_index": source})
