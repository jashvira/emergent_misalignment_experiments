"""Helpers for C/C++ function-source datasets."""

from __future__ import annotations

import hashlib
import re


CONTROL_WORDS = {"if", "for", "while", "switch", "return", "sizeof", "catch"}


def normalize_code(code: str) -> str:
    code = re.sub(r"/\*.*?\*/", "", code or "", flags=re.S)
    code = re.sub(r"//.*", "", code)
    return re.sub(r"\s+", "", code).casefold()


def stable_code_hash(code: str) -> str:
    return hashlib.sha256(normalize_code(code).encode("utf-8")).hexdigest()


def extract_function_name(code: str) -> str:
    pending: list[str] = []
    for raw_line in (code or "").splitlines():
        line = raw_line.strip()
        if not line:
            pending = []
            continue
        if line.startswith(("#", "//", "/*", "*")):
            continue
        pending.append(line)
        if "{" not in line:
            if ";" in line:
                pending = []
            continue
        header = " ".join(pending).split("{", 1)[0].strip()
        pending = []
        if "(" not in header or ")" not in header or "=" in header:
            continue
        before_paren = header.split("(", 1)[0].strip()
        before_paren = re.sub(r"<[^<>]*>", "", before_paren)
        tokens = before_paren.replace("*", " ").replace("&", " ").split()
        if not tokens:
            continue
        name = tokens[-1].split("::")[-1]
        if name not in CONTROL_WORDS:
            return name
    return "unknown"


def best_function_name(*codes: str) -> str:
    for code in codes:
        name = extract_function_name(code)
        if name != "unknown":
            return name
    return "unknown"
