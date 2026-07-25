"""validate_frame_map.py — Validate frame_map.json khớp với FAISS index.

Nhiệm vụ:
  1. Đọc frame_map.json
  2. Load FAISS index để lấy ntotal
  3. Kiểm tra số lượng, faiss_indices liên tục, required fields
  4. Bổ sung timestamp_source / timestamp_confidence nếu thiếu
  5. Xuất báo cáo JSON

Usage:
    python backend/app/services/indexing/validate_frame_map.py \\
        --frame-map data/metadata/siglip2_so400m_patch16_384_frame_map.json \\
        --faiss-index data/indexes/siglip2_so400m_patch16_384_flat_ip.faiss \\
        --report data/metadata/frame_map_validation_report.json \\
        [--fix]  # Tự động bổ sung timestamp_source/confidence nếu thiếu
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

try:
    import faiss
except ImportError as exc:
    raise SystemExit(
        "FAISS is required. Install with: pip install -r requirements.txt"
    ) from exc

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {"frame_id", "video_id", "timestamp", "keyframe_path"}


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def infer_timestamp_source(record: dict) -> str:
    if record.get("frame_index") is not None:
        return "video_fps"
    if record.get("timestamp") is not None:
        return "interval"
    return "unknown"


def infer_timestamp_confidence(record: dict) -> float:
    source = record.get("timestamp_source") or infer_timestamp_source(record)
    return {"video_fps": 1.0, "matched_frame": 0.9, "interval": 0.5, "unknown": 0.0}.get(
        source, 0.5
    )


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------

def load_frame_map(path: Path) -> dict[int, dict]:
    with path.open("r", encoding="utf-8") as f:
        raw: dict = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"frame_map.json phải là dict, got {type(raw).__name__}")

    result: dict[int, dict] = {}
    for key, value in raw.items():
        try:
            idx = int(key)
        except (ValueError, TypeError):
            logger.warning("Bỏ qua key không phải số: %r", key)
            continue
        result[idx] = value
    return result


def validate_frame_map(
    frame_map: dict[int, dict],
    faiss_ntotal: int,
    fix: bool = False,
) -> tuple[dict, dict[int, dict]]:
    """Validate frame_map so với FAISS ntotal.

    Args:
        frame_map: Dict[faiss_index -> record]
        faiss_ntotal: index.ntotal
        fix: Nếu True, tự động bổ sung timestamp_source/confidence

    Returns:
        (report dict, possibly-fixed frame_map)
    """
    errors = []
    warnings = []
    fixes_applied = []
    stored_count = len(frame_map)

    # 1. Kiểm tra số lượng
    if stored_count != faiss_ntotal:
        errors.append(
            f"COUNT_MISMATCH: frame_map={stored_count} records vs FAISS={faiss_ntotal} vectors"
        )

    # 2. Kiểm tra faiss_indices liên tục 0..N-1
    expected = set(range(faiss_ntotal))
    actual = set(frame_map.keys())
    missing_indices = sorted(expected - actual)
    extra_indices = sorted(actual - expected)

    if missing_indices:
        sample = missing_indices[:20]
        errors.append(
            f"MISSING_INDICES: {len(missing_indices)} faiss_index thiếu trong frame_map "
            f"(sample: {sample})"
        )

    if extra_indices:
        sample = extra_indices[:20]
        warnings.append(
            f"EXTRA_INDICES: {len(extra_indices)} faiss_index trong frame_map nằm ngoài FAISS range "
            f"(sample: {sample})"
        )

    # 3. Per-record validation
    schema_errors = []
    encoder_contract_warnings = []
    missing_timestamp_source = []
    fixed_frame_map = dict(frame_map)

    for idx in sorted(frame_map.keys()):
        record = dict(frame_map[idx])  # copy để fix

        # Required fields
        missing_fields = [f for f in REQUIRED_FIELDS if not record.get(f) and record.get(f) != 0]
        empty_fields = [f for f in REQUIRED_FIELDS if f in record and record[f] in ("", None)]
        if missing_fields or empty_fields:
            schema_errors.append({
                "faiss_index": idx,
                "frame_id": record.get("frame_id"),
                "missing_fields": missing_fields,
                "empty_fields": empty_fields,
            })

        missing_encoder_fields = [
            field
            for field in ("model_name", "model_revision", "vector_dim")
            if record.get(field) in ("", None)
        ]
        if missing_encoder_fields:
            encoder_contract_warnings.append(
                {
                    "faiss_index": idx,
                    "frame_id": record.get("frame_id"),
                    "missing_fields": missing_encoder_fields,
                }
            )

        # timestamp_source / timestamp_confidence
        needs_fix = False
        if not record.get("timestamp_source"):
            source = infer_timestamp_source(record)
            if fix:
                record["timestamp_source"] = source
                needs_fix = True
            else:
                missing_timestamp_source.append(idx)

        if record.get("timestamp_confidence") is None:
            conf = infer_timestamp_confidence(record)
            if fix:
                record["timestamp_confidence"] = conf
                needs_fix = True

        if fix and needs_fix:
            fixed_frame_map[idx] = record
            fixes_applied.append(idx)

    if schema_errors:
        errors.append(
            f"SCHEMA_ERRORS: {len(schema_errors)} records thiếu required fields"
        )

    if missing_timestamp_source and not fix:
        warnings.append(
            f"MISSING_TIMESTAMP_SOURCE: {len(missing_timestamp_source)} records thiếu "
            f"'timestamp_source' field. Chạy với --fix để tự động bổ sung."
        )

    if encoder_contract_warnings:
        warnings.append(
            f"MISSING_ENCODER_CONTRACT: {len(encoder_contract_warnings)} records are "
            "missing model_name, model_revision, or vector_dim. Legacy frame maps remain "
            "readable, but new SigLIP2 indexes must include these fields."
        )

    # 4. Timestamp source summary
    source_counts: dict[str, int] = {}
    for rec in fixed_frame_map.values():
        src = rec.get("timestamp_source") or "missing"
        source_counts[src] = source_counts.get(src, 0) + 1

    report = {
        "status": "passed" if not errors else "failed",
        "warnings": warnings,
        "errors": errors,
        "checks": {
            "count_match": stored_count == faiss_ntotal,
            "no_missing_indices": not missing_indices,
            "no_extra_indices": not extra_indices,
            "no_schema_errors": not schema_errors,
        },
        "stats": {
            "frame_map_count": stored_count,
            "faiss_ntotal": faiss_ntotal,
            "missing_indices_count": len(missing_indices),
            "extra_indices_count": len(extra_indices),
            "schema_error_count": len(schema_errors),
            "schema_errors_sample": schema_errors[:10],
            "missing_timestamp_source_count": len(missing_timestamp_source),
            "encoder_contract_warning_count": len(encoder_contract_warnings),
            "encoder_contract_warnings_sample": encoder_contract_warnings[:10],
            "fixes_applied_count": len(fixes_applied),
            "timestamp_source_distribution": source_counts,
        },
    }

    return report, fixed_frame_map


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate frame_map.json khớp với FAISS index."
    )
    parser.add_argument(
        "--frame-map",
        type=Path,
        default=Path("data/metadata/siglip2_so400m_patch16_384_frame_map.json"),
        help="Đường dẫn tới frame_map.json",
    )
    parser.add_argument(
        "--faiss-index",
        type=Path,
        default=Path("data/indexes/siglip2_so400m_patch16_384_flat_ip.faiss"),
        help="Đường dẫn tới FAISS index file (.faiss)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "data/metadata/siglip2_so400m_patch16_384_frame_map_validation.json"
        ),
        help="Đường dẫn xuất báo cáo JSON",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Tự động bổ sung timestamp_source/confidence nếu thiếu và ghi lại frame_map",
    )
    parser.add_argument(
        "--fixed-frame-map",
        type=Path,
        default=None,
        help="Nếu --fix, ghi frame_map đã fix ra đây (mặc định overwrite frame_map gốc)",
    )
    return parser.parse_args()


def main() -> None:
    started_at = time.perf_counter()
    args = parse_args()

    # Load frame_map
    if not args.frame_map.exists():
        raise SystemExit(f"frame_map.json không tồn tại: {args.frame_map}")
    logger.info("Loading frame_map: %s", args.frame_map)
    frame_map = load_frame_map(args.frame_map)
    logger.info("frame_map loaded: %d records", len(frame_map))

    # Load FAISS index
    if not args.faiss_index.exists():
        raise SystemExit(f"FAISS index không tồn tại: {args.faiss_index}")
    logger.info("Loading FAISS index: %s", args.faiss_index)
    index = faiss.read_index(str(args.faiss_index))
    faiss_ntotal = int(index.ntotal)
    logger.info("FAISS index loaded: ntotal=%d, dim=%d", faiss_ntotal, index.d)

    # Validate
    report, fixed_frame_map = validate_frame_map(frame_map, faiss_ntotal, fix=args.fix)
    report["runtime_sec"] = round(time.perf_counter() - started_at, 3)
    report["frame_map_path"] = str(args.frame_map)
    report["faiss_index_path"] = str(args.faiss_index)

    # Ghi report
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("Validation report: %s", args.report)

    # Ghi frame_map đã fix nếu có
    if args.fix and report["stats"]["fixes_applied_count"] > 0:
        out_path = args.fixed_frame_map or args.frame_map
        str_keyed = {str(k): v for k, v in fixed_frame_map.items()}
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(str_keyed, f, ensure_ascii=False, indent=2)
        logger.info(
            "Đã ghi frame_map đã fix (%d bản ghi) → %s",
            report["stats"]["fixes_applied_count"],
            out_path,
        )

    # Summary
    status = report["status"].upper()
    print(f"\n{'='*50}")
    print(f"  Status: {status}")
    print(f"  frame_map records: {report['stats']['frame_map_count']}")
    print(f"  FAISS ntotal:      {faiss_ntotal}")
    print(f"  Missing indices:   {report['stats']['missing_indices_count']}")
    print(f"  Schema errors:     {report['stats']['schema_error_count']}")
    print(f"  Fixes applied:     {report['stats']['fixes_applied_count']}")
    print(f"  timestamp_source:  {report['stats']['timestamp_source_distribution']}")
    print(f"{'='*50}\n")

    if report["errors"]:
        for err in report["errors"]:
            logger.error(err)
    if report["warnings"]:
        for w in report["warnings"]:
            logger.warning(w)

    if report["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
