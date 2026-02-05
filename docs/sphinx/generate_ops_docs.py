#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[2]
OPS_JSON = ROOT / "ops.json"
OUT_DIR = ROOT / "docs" / "sphinx" / "ops"
CAT_DIR = OUT_DIR / "categories"
OP_DIR = OUT_DIR / "operations"


def ensure_dirs() -> None:
    CAT_DIR.mkdir(parents=True, exist_ok=True)
    OP_DIR.mkdir(parents=True, exist_ok=True)


def load_ops() -> Dict[str, Any]:
    with OPS_JSON.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sort_ops(ops: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(ops, key=lambda op: op["name"])


def render_ops_index(ops: List[Dict[str, Any]]) -> str:
    lines = [
        "Operations Catalog",
        "==================",
        "",
        "This section is generated from `ops.json` to stay in sync with the runtime.",
        "",
        "Summary",
        "-------",
        f"- total ops: {len(ops)}",
        "",
        "Alphabetical Index",
        "------------------",
        "",
        ".. list-table::",
        "   :header-rows: 1",
        "",
        "   * - Op",
        "     - Category",
        "     - Capabilities",
        "     - Devices",
    ]
    for op in ops:
        caps = []
        if op.get("broadcast") == "allow":
            caps.append("broadcast")
        if op.get("inplace") == "allow":
            caps.append("inplace")
        if op.get("accumulate") == "allow":
            caps.append("accumulate")
        cap_str = ", ".join(caps) if caps else "-"
        devices = op.get("devices", {})
        dev_str = ", ".join([k for k, v in devices.items() if v]) if devices else "-"
        lines.append(f"   * - :doc:`operations/{op['name']}`")
        lines.append(f"     - {op['category']}")
        lines.append(f"     - {cap_str}")
        lines.append(f"     - {dev_str}")
    lines += [
        "",
        "By Capability",
        "-------------",
        "",
        "- broadcast: ops that allow shape broadcasting",
        "- inplace: ops that can write into an existing output buffer",
        "- accumulate: ops that support accumulation modes",
        "",
        "Device Matrix",
        "-------------",
        "",
        ".. list-table::",
        "   :header-rows: 1",
        "",
        "   * - Op",
        "     - CPU",
        "     - Vulkan",
    ]
    for op in ops:
        devices = op.get("devices", {})
        cpu = "yes" if devices.get("cpu") else "no"
        vk = "yes" if devices.get("vulkan") else "no"
        lines.append(f"   * - :doc:`operations/{op['name']}`")
        lines.append(f"     - {cpu}")
        lines.append(f"     - {vk}")
    lines += [
        "",
        "By Device",
        "---------",
        "",
        "Ops can declare CPU and/or Vulkan support in `ops.json`.",
        "",
        "By Category",
        "-----------",
        "",
        ".. toctree::",
        "   :maxdepth: 1",
        "   :glob:",
        "",
        "   categories/*",
        "",
        "By Operation",
        "------------",
        "",
        ".. toctree::",
        "   :maxdepth: 1",
        "   :glob:",
        "",
        "   operations/*",
        "",
    ]
    return "\n".join(lines) + "\n"


def render_category_page(category: str, ops: List[Dict[str, Any]]) -> str:
    device_counts = {"cpu": 0, "vulkan": 0}
    for op in ops:
        devices = op.get("devices", {})
        for device, enabled in devices.items():
            if enabled and device in device_counts:
                device_counts[device] += 1
    lines = [
        f"{category.title()} Operations",
        "=" * (len(category) + 11),
        "",
        f"Operations in the `{category}` category.",
        "",
        f"- total ops: {len(ops)}",
        f"- cpu support: {device_counts['cpu']}",
        f"- vulkan support: {device_counts['vulkan']}",
        "",
        ".. list-table::",
        "   :header-rows: 1",
        "",
        "   * - Op",
        "     - Inputs",
        "     - Outputs",
        "     - Capabilities",
    ]
    for op in ops:
        inputs = op["inputs"]["count"] if op["inputs"]["arity"] == "fixed" else op["inputs"]["arity"]
        outputs = op["outputs"]["count"] if op["outputs"]["arity"] == "fixed" else op["outputs"]["arity"]
        caps = []
        if op.get("broadcast") == "allow":
            caps.append("broadcast")
        if op.get("inplace") == "allow":
            caps.append("inplace")
        if op.get("accumulate") == "allow":
            caps.append("accumulate")
        cap_str = ", ".join(caps) if caps else "-"
        lines.append(f"   * - :doc:`../operations/{op['name']}`")
        lines.append(f"     - {inputs}")
        lines.append(f"     - {outputs}")
        lines.append(f"     - {cap_str}")
    return "\n".join(lines) + "\n"


def render_op_page(op: Dict[str, Any], dtype_sets: Dict[str, Any]) -> str:
    name = op["name"]
    category = op["category"]
    inputs = op["inputs"]
    outputs = op["outputs"]
    attrs = op.get("attrs", [])
    broadcast = op.get("broadcast", "deny")
    inplace = op.get("inplace", "deny")
    accumulate = op.get("accumulate", "deny")
    devices = op.get("devices", {})
    dtype_ref = op.get("dtype_support_ref")
    dtype_support = dtype_sets.get(dtype_ref, {})
    normal = ", ".join(dtype_support.get("normal", [])) or "-"
    accumulate_pairs = dtype_support.get("accumulate", [])
    acc_str = ", ".join([f"{p['input']}→{p['acc']}" for p in accumulate_pairs]) or "-"
    input_arity = (
        str(inputs["count"]) if inputs["arity"] == "fixed" else inputs["arity"]
    )
    output_arity = (
        str(outputs["count"]) if outputs["arity"] == "fixed" else outputs["arity"]
    )
    devices_str = ", ".join([k for k, v in devices.items() if v]) or "-"
    attrs_str = ", ".join(attrs) if attrs else "none"

    lines = [
        name,
        "=" * len(name),
        "",
        f"**Category:** {category}",
        "",
        "Signature",
        "---------",
        f"`op {name}(...) >> out`",
        "",
        "Arity",
        "-----",
        f"- inputs: {input_arity}",
        f"- outputs: {output_arity}",
        "",
        "Attributes",
        "----------",
        f"- {attrs_str}",
        "",
        "Capabilities",
        "------------",
        f"- broadcast: {broadcast}",
        f"- inplace: {inplace}",
        f"- accumulate: {accumulate}",
        "",
        "Devices",
        "-------",
        f"- {devices_str}",
        "",
        "DTypes",
        "------",
        f"- normal: {normal}",
        f"- accumulate: {acc_str}",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ensure_dirs()
    data = load_ops()
    ops = sort_ops(data["ops"])
    dtype_sets = data.get("dtype_sets", {})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "index.rst").write_text(render_ops_index(ops), encoding="utf-8")

    by_category: Dict[str, List[Dict[str, Any]]] = {}
    for op in ops:
        by_category.setdefault(op["category"], []).append(op)

    for category, cat_ops in sorted(by_category.items()):
        (CAT_DIR / f"{category}.rst").write_text(
            render_category_page(category, cat_ops), encoding="utf-8"
        )

    for op in ops:
        (OP_DIR / f"{op['name']}.rst").write_text(
            render_op_page(op, dtype_sets), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
