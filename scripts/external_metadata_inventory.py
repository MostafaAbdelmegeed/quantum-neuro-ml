"""Snapshot public ABIDE II metadata and audit candidate participant identities."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from quantum_neuro_ml.provenance import digest, recorded_run


def canonical_id(value: str) -> int:
    match = re.fullmatch(r"(?:sub-)?(\d+)", str(value).strip(), re.IGNORECASE)
    if not match or int(match.group(1)) <= 0:
        raise ValueError("Invalid canonical participant identifier")
    return int(match.group(1))


def decode_metadata(payload: bytes) -> tuple[str, str]:
    try:
        return payload.decode("utf-8-sig"), "utf-8-sig"
    except UnicodeDecodeError:
        if {value for value in payload if value >= 128} <= {160}:
            return payload.decode("latin-1"), "latin1_nbsp_only"
        raise


def parse_table(payload: bytes, delimiter: str = "\t", *, allow_row_index: bool = False) -> tuple[list[str], list[dict]]:
    reader = csv.reader(io.StringIO(decode_metadata(payload)[0]), delimiter=delimiter)
    fields = [name.strip().lower() for name in next(reader)]
    has_row_index = bool(allow_row_index and fields and fields[0] == "")
    if has_row_index:
        fields[0] = "__row_index"
    if not all(fields) or len(fields) != len(set(fields)):
        raise ValueError("Empty or duplicate metadata column after trimming headers")
    rows = []
    for values in reader:
        if not values or not any(value.strip() for value in values):
            continue
        if len(values) != len(fields):
            raise ValueError("Metadata row width differs from its header")
        rows.append(dict(zip(fields, [value.strip() for value in values])))
    if has_row_index and any(row["__row_index"] != str(index) for index, row in enumerate(rows)):
        raise ValueError("Unnamed development column is not the expected sequential row index")
    return fields, rows


def guid(value: str) -> str:
    value = str(value).strip().upper()
    return "" if value in ("", "N/A", "NA", "NAN", "-9999", "NONE") else value


def number(value: str) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else float("nan")
    except (ValueError, TypeError):
        return float("nan")


def candidate_rows(collections: dict, development_rows: list[dict], excluded: set[str]) -> list[dict]:
    development_ids = {canonical_id(row["sub_id"]) for row in development_rows}
    development_guids = {guid(row.get("ndar_guid", "")) for row in development_rows} - {""}
    records = []
    for collection, (_, rows) in sorted(collections.items()):
        for source in rows:
            reasons = []
            try:
                identity = canonical_id(source.get("participant_id", ""))
            except ValueError:
                identity = None
                reasons.append("invalid_participant_id")
            source_guid = guid(source.get("ndar_guid", ""))
            if collection in excluded:
                reasons.append("known_longitudinal_collection")
            if identity in development_ids:
                reasons.append("abide1_numeric_identity_overlap")
            if source_guid and source_guid in development_guids:
                reasons.append("abide1_ndar_guid_overlap")
            site = source.get("site_id", "").removeprefix("ABIDEII-").upper()
            if site != collection.removeprefix("ABIDEII-").upper():
                reasons.append("site_identifier_mismatch")
            diagnosis, sex, age = number(source.get("dx_group", "")), number(source.get("sex", "")), number(source.get("age_at_scan", ""))
            if diagnosis not in (1, 2):
                reasons.append("missing_or_invalid_diagnosis")
            if sex not in (1, 2):
                reasons.append("missing_or_invalid_sex")
            if not age > 0:
                reasons.append("missing_or_invalid_age")
            records.append({"collection": collection, "source_participant_id": source.get("participant_id", ""),
                            "canonical_id": identity, "diagnosis": int(diagnosis) if diagnosis in (1, 2) else None,
                            "age": age if math.isfinite(age) else None, "sex": int(sex) if sex in (1, 2) else None,
                            "ndar_guid_sha256": hashlib.sha256(source_guid.encode()).hexdigest() if source_guid else "",
                            "reasons": reasons})
    ids = Counter(row["canonical_id"] for row in records if row["canonical_id"] is not None)
    guids = Counter(row["ndar_guid_sha256"] for row in records if row["ndar_guid_sha256"])
    for row in records:
        if row["canonical_id"] is not None and ids[row["canonical_id"]] > 1:
            row["reasons"].append("ambiguous_repeated_numeric_identity")
        if row["ndar_guid_sha256"] and guids[row["ndar_guid_sha256"]] > 1:
            row["reasons"].append("ambiguous_repeated_ndar_guid")
        row["metadata_candidate"] = not row["reasons"]
    candidate_ids = {row["canonical_id"] for row in records if row["metadata_candidate"]}
    if candidate_ids & development_ids:
        raise ValueError("Development identity survived the external overlap exclusion")
    return records


def cached_fetch(url: str, path: Path, offline: bool) -> dict:
    receipt_path = path.with_suffix(path.suffix + ".receipt.json")
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["url"] != url:
            raise ValueError("Cached metadata URL differs")
        if receipt["status"] == 404:
            return receipt
        if not path.is_file() or digest(path) != receipt["sha256"]:
            raise ValueError("Cached metadata checksum differs")
        return receipt
    if offline:
        raise ValueError("Metadata cache is incomplete for offline use")
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"url": url, "retrieved_utc": datetime.now(timezone.utc).isoformat()}
    try:
        with urllib.request.urlopen(url, timeout=45) as response:
            payload = response.read(8_000_001)
            if len(payload) > 8_000_000 or (response.headers.get("Content-Length") and len(payload) != int(response.headers["Content-Length"])):
                raise ValueError("Metadata response was oversized or truncated")
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_bytes(payload)
        temporary.replace(path)
        receipt.update(status=200, bytes=len(payload), sha256=digest(path))
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        receipt.update(status=404)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def validate_captured_inputs(run, config: dict, configuration_payload: bytes, script_payload: bytes, sources: dict) -> None:
    captured = run.record["inputs"]
    if captured["development_metadata"]["sha256"] != config["abide1_metadata_sha256"]:
        raise ValueError("Captured development metadata differs from the frozen identity snapshot")
    for name, payload in (("configuration", configuration_payload), ("runner_script", script_payload)):
        if captured[name]["sha256"] != hashlib.sha256(payload).hexdigest():
            raise ValueError("Captured inventory configuration or script differs")
    for name, source in sources.items():
        if source["status"] == 200 and captured[name]["sha256"] != source["sha256"]:
            raise ValueError(f"Captured source differs from its retrieval receipt: {name}")
        if json.loads(run.input_payloads[name + "_receipt"]) != source:
            raise ValueError(f"Captured retrieval receipt differs: {name}")


def run(config_path: Path, development_path: Path, cache: Path, output: Path, offline: bool) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Inventory outputs already exist; choose a new directory")
    script_payload = Path(__file__).read_bytes()
    configuration_payload = config_path.read_bytes()
    config = json.loads(configuration_payload)
    if (config["version"] != "abide2-metadata-inventory-v1" or config["model_eligible"] is not False
            or config["bucket_url"] != "https://fcp-indi.s3.amazonaws.com/"
            or config["raw_prefix"] != "data/Projects/ABIDE2/RawData/"
            or config["excluded_collections"] != ["ABIDEII-UCLA_Long", "ABIDEII-UPSM_Long"]
            or config["required_fields"] != ["participant_id", "site_id", "dx_group", "age_at_scan", "sex"]
            or config["text_encoding"] != "utf8_sig_or_latin1_for_isolated_nbsp_only"
            or config["duplicate_identity_policy"] != "exclude_all_ambiguous_canonical_ids_and_nonmissing_ndar_guids"
            or config["candidate_meaning"] != "metadata_complete_and_no_detected_development_release_overlap"):
        raise ValueError("Unsupported external inventory configuration")
    if digest(development_path) != config["abide1_metadata_sha256"]:
        raise ValueError("Development metadata differs from the frozen ABIDE I snapshot")
    index = cache / "raw-index.xml"
    query = urllib.parse.urlencode({"list-type": "2", "prefix": config["raw_prefix"], "delimiter": "/"})
    sources = {"raw_index": cached_fetch(config["bucket_url"] + "?" + query, index, offline)}
    tree = ET.fromstring(index.read_bytes())
    if tree.findtext("{*}IsTruncated") != "false":
        raise ValueError("Public collection inventory is truncated")
    prefixes = [p.text for p in tree.findall(".//{*}CommonPrefixes/{*}Prefix")]
    selected = [p for p in prefixes if p.rstrip("/").split("/")[-1].startswith("ABIDEII-")]
    paths = {}
    tasks = []
    for prefix in selected:
        collection = prefix.rstrip("/").split("/")[-1]
        for filename in ("participants.tsv", "dataset_description.json"):
            name = f"{collection}/{filename}"
            path = cache / name
            paths[name] = path
            tasks.append((name, config["bucket_url"] + prefix + filename, path))
    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(lambda task: (task[0], cached_fetch(task[1], task[2], offline)), tasks))
    sources.update(outcomes)
    if config_path.read_bytes() != configuration_payload or Path(__file__).read_bytes() != script_payload:
        raise ValueError("Inventory configuration or script changed during metadata retrieval")
    output.mkdir(parents=True, exist_ok=True)
    entrypoint = output / "entrypoint.json"
    entrypoint.write_text(json.dumps({"script_sha256": hashlib.sha256(script_payload).hexdigest(),
                                      "script_utf8": script_payload.decode("utf-8")}), encoding="utf-8")
    inputs = {"configuration": config_path, "development_metadata": development_path,
              "raw_index": index, "raw_index_receipt": index.with_suffix(index.suffix + ".receipt.json"),
              "runner_script": Path(__file__), "entrypoint_snapshot": entrypoint}
    for name, path in paths.items():
        if sources[name]["status"] == 200:
            inputs[name] = path
        inputs[name + "_receipt"] = path.with_suffix(path.suffix + ".receipt.json")
    with recorded_run(output, {"experiment": config["version"], "offline": offline}, inputs) as receipt:
        validate_captured_inputs(receipt, config, configuration_payload, script_payload, sources)
        _, development = parse_table(development_path.read_bytes(), ",", allow_row_index=True)
        collections, missing, descriptions = {}, [], {}
        for prefix in selected:
            collection = prefix.rstrip("/").split("/")[-1]
            name = f"{collection}/participants.tsv"
            if sources[name]["status"] != 200:
                missing.append(collection)
                continue
            fields, rows = parse_table(paths[name].read_bytes())
            if not set(config["required_fields"]) <= set(fields):
                raise ValueError(f"Required external metadata columns are missing in {collection}")
            collections[collection] = (fields, rows)
            description_name = f"{collection}/dataset_description.json"
            if sources[description_name]["status"] == 200:
                descriptions[collection] = json.loads(paths[description_name].read_text(encoding="utf-8-sig")).get("License")
        records = candidate_rows(collections, development, set(config["excluded_collections"]))
        fieldnames = ["collection", "source_participant_id", "canonical_id", "diagnosis", "age", "sex", "ndar_guid_sha256", "metadata_candidate", "reasons"]
        with (output / "candidates.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows({**row, "reasons": ";".join(row["reasons"])} for row in records)
        v1_qc = ["func_mean_fd", "qc_rater_1", "qc_anat_rater_2", "qc_func_rater_2", "qc_anat_rater_3", "qc_func_rater_3"]
        summary = {}
        for collection, (fields, rows) in collections.items():
            candidates = [r for r in records if r["collection"] == collection and r["metadata_candidate"]]
            summary[collection] = {"source_rows": len(rows), "metadata_candidates": len(candidates),
                                   "metadata_encoding": decode_metadata(paths[f"{collection}/participants.tsv"].read_bytes())[1],
                                   "candidate_diagnoses": dict(Counter("ASD" if r["diagnosis"] == 1 else "control" for r in candidates)),
                                   "missing_v1_qc_columns": [name for name in v1_qc if name not in fields],
                                   "declared_license": descriptions.get(collection)}
        report = {"purpose": "external_metadata_candidate_inventory", "configuration": config,
                  "development_identities": len({canonical_id(r["sub_id"]) for r in development}),
                  "listed_collections": len(selected), "missing_metadata_collections": missing,
                  "metadata_inventory_complete_for_listed_collections": not missing,
                  "source_rows": len(records), "metadata_candidates": sum(r["metadata_candidate"] for r in records),
                  "detected_abide1_numeric_overlap_rows": sum("abide1_numeric_identity_overlap" in r["reasons"] for r in records),
                  "known_longitudinal_rows": sum("known_longitudinal_collection" in r["reasons"] for r in records),
                  "exclusion_reason_counts": dict(Counter(reason for r in records for reason in r["reasons"])),
                  "candidate_development_id_overlap": 0, "model_eligible": False,
                  "identity_check_scope": "canonical_numeric_ids_and_available_ndar_guids_not_a_complete_person_crosswalk",
                  "development_guid_crosswalk_available": any(guid(r.get("ndar_guid", "")) for r in development),
                  "external_nonmissing_guid_rows": sum(bool(r["ndar_guid_sha256"]) for r in records),
                  "unresolved_gates": ["official_release_coverage_and_corrections", "compatible_CC200_preprocessing", "motion_and_manual_QC", "collection_terms_and_acknowledgements"],
                  "by_collection": summary, "source_receipts": sources,
                  "candidate_manifest_sha256": digest(output / "candidates.csv")}
        result = receipt.finish(report)
        print(json.dumps({key: result[key] for key in ("source_rows", "metadata_candidates", "known_longitudinal_rows", "detected_abide1_numeric_overlap_rows", "model_eligible")}))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/abide2-inventory-v1.json"))
    parser.add_argument("--development-metadata", type=Path, default=Path("data/metadata/Phenotypic_V1_0b_preprocessed1.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/external/abide2-metadata-v1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    run(args.config, args.development_metadata, args.cache_dir, args.output_dir, args.offline)


if __name__ == "__main__":
    main()
