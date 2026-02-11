import gzip
import shlex

import numpy as np

from typing import Any, Dict, List, Tuple


class TypedStreamError(ValueError):
    pass


def read_typedstream(path: str) -> Tuple[Dict[str, Any], str]:
    """Read a CMTK TypedStream file into a nested dict.

    Returns (data, version).
    """
    open_func = gzip.open if path.endswith(".gz") else open
    with open_func(path, "rt") as handle:
        header = handle.readline().strip()
        if not header.startswith("! TYPEDSTREAM"):
            raise TypedStreamError(f"Not a CMTK TypedStream: {path}")
        version = header.split(" ")[-1]
        lines = handle.readlines()

    data, _ = _parse_block(lines, 0)
    return data, version


def _parse_block(lines: List[str], idx: int) -> Tuple[Dict[str, Any], int]:
    data: Dict[str, Any] = {}

    while idx < len(lines):
        line = lines[idx].strip()
        idx += 1
        if not line:
            continue

        tokens = shlex.split(line)
        if not tokens:
            continue

        label = tokens[0]
        if label == "}":
            return data, idx

        if tokens[-1] == "{":
            sub, idx = _parse_block(lines, idx)
            _insert_item(data, label, sub)
            continue

        if label == "coefficients":
            if "dims" not in data:
                raise TypedStreamError("Coefficients found before dims")
            num = int(np.prod(data["dims"])) * 3
            values = _parse_numeric_tokens(tokens[1:])
            while len(values) < num and idx < len(lines):
                line = lines[idx].strip()
                if line == "}":
                    break
                idx += 1
                if not line:
                    continue
                more_tokens = shlex.split(line)
                values.extend(_parse_numeric_tokens(more_tokens))
            if len(values) != num:
                raise TypedStreamError(
                    f"Coefficient count mismatch: got {len(values)}, expected {num}"
                )
            data[label] = np.array(values, dtype=float).reshape(-1, 3)
            continue

        if label == "active":
            if "dims" not in data:
                raise TypedStreamError("Active found before dims")
            num = int(np.prod(data["dims"])) * 3
            bit_str = ""
            while len(bit_str) < num and idx < len(lines):
                line = lines[idx].strip()
                if line == "}":
                    break
                idx += 1
                if not line:
                    continue
                bit_str += line
            # Pad with zeros if necessary (some files may have incomplete active fields)
            if len(bit_str) < num:
                bit_str += "0" * (num - len(bit_str))
            data[label] = np.array([int(b) for b in bit_str[:num]], dtype=np.uint8)
            continue

        items = tokens[1:]
        value = _parse_value(items)
        _insert_item(data, label, value)

    return data, idx


def _parse_numeric_tokens(tokens: List[str]) -> List[float]:
    return [float(t) for t in tokens]


def _parse_value(tokens: List[str]) -> Any:
    if not tokens:
        return None
    if tokens[0] in ("yes", "no"):
        return tokens[0] == "yes"
    if _is_number(tokens[0]):
        nums = [float(t) for t in tokens]
        if len(nums) == 1:
            return nums[0]
        return nums
    if len(tokens) == 1:
        return tokens[0]
    return tokens


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _insert_item(data: Dict[str, Any], key: str, value: Any) -> None:
    if key not in data:
        data[key] = value
        return
    current = data[key]
    if isinstance(current, list):
        current.append(value)
        return
    data[key] = [current, value]
