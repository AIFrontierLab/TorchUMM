"""Relocate fixed training manifests without changing their supervision."""

from __future__ import annotations

import json
from pathlib import Path


def remap_reference(reference, bindings):
    result = dict(reference)
    for key in ("path", "shard_path"):
        if key in result:
            value = str(result[key])
            for old, new in sorted(bindings.items(), key=lambda item: -len(item[0])):
                if value == old or value.startswith(old.rstrip("/") + "/"):
                    result[key] = new.rstrip("/") + value[len(old.rstrip("/")) :]
                    break
    if "image_ref" in result:
        result["image_ref"] = remap_reference(result["image_ref"], bindings)
    return result


def check_reference(reference):
    kind = reference.get("type")
    if kind in ("file", "local_file", "webdataset_tar_member"):
        path = Path(reference["shard_path" if kind == "webdataset_tar_member" else "path"])
        if not path.is_file():
            raise FileNotFoundError(path)
    elif kind == "image_crop":
        if len(reference.get("bbox", ())) != 4:
            raise ValueError("A crop needs four pixel coordinates")
        check_reference(reference["image_ref"])
    else:
        raise ValueError(f"Unsupported image reference: {kind}")


def prepare_data(t2i_manifest, i2t_manifest, path_map, output):
    bindings = json.loads(Path(path_map).read_text(encoding="utf-8"))
    if not isinstance(bindings, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in bindings.items()
    ):
        raise TypeError("Path map must be a JSON object of source-prefix: destination-prefix")
    output = Path(output).expanduser().resolve()
    prepared, provenance = {}, {}
    for route, source, expected in (("t2i", t2i_manifest, 2460), ("i2t", i2t_manifest, 4000)):
        source = Path(source)
        data = source.read_bytes()
        records = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
        if len(records) != expected or len({r["sample_id"] for r in records}) != expected:
            raise ValueError(f"{route} requires the released {expected} unique records")
        rows = []
        for record in records:
            row = dict(record)
            for key in ("image_ref", "local_crop_ref"):
                if isinstance(row.get(key), dict):
                    row[key] = remap_reference(row[key], bindings)
            check_reference(row["image_ref"])
            if route == "t2i":
                if not row.get("local_phrase") or not isinstance(row.get("local_crop_ref"), dict):
                    raise ValueError(f"Missing grounded supervision: {row['sample_id']}")
                check_reference(row["local_crop_ref"])
            rows.append(row)
        prepared[route] = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        provenance[route] = {"records": expected}
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for route, content in prepared.items():
        (output / f"{route}.jsonl").write_text(content, encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
