"""Exact and near duplicates at scale.

Exact: sha256 (the store already makes that a no-op). Near: 64-bit dHash on a
9x8 grey thumbnail. Near-duplicates within Hamming distance ``t`` are found by
splitting the hash into ``t+1`` bands: two hashes within distance ``t`` must
agree on at least one band (pigeonhole), so candidates come from band buckets
and only candidates are compared. Linear in the corpus, no pairwise scan.
"""

from __future__ import annotations

import io
from collections import defaultdict

import numpy as np
from PIL import Image


def dhash(data: bytes | Image.Image, size: int = 8) -> int:
    im = data if isinstance(data, Image.Image) else Image.open(io.BytesIO(data))
    g = np.asarray(im.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def near_duplicates(hashes: dict[str, int], max_distance: int = 3, bits: int = 64) -> list[tuple[str, str, int]]:
    """Pairs (key_a, key_b, distance) with distance <= max_distance, each pair once."""
    bands = max_distance + 1
    width = bits // bands
    buckets: list[dict[int, list[str]]] = [defaultdict(list) for _ in range(bands)]
    for key, h in hashes.items():
        for b in range(bands):
            chunk = (h >> (b * width)) & ((1 << width) - 1)
            buckets[b][chunk].append(key)
    seen: set[tuple[str, str]] = set()
    out = []
    for b in range(bands):
        for members in buckets[b].values():
            if len(members) < 2:
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    pair = (members[i], members[j]) if members[i] < members[j] else (members[j], members[i])
                    if pair in seen:
                        continue
                    seen.add(pair)
                    d = hamming(hashes[pair[0]], hashes[pair[1]])
                    if d <= max_distance:
                        out.append((pair[0], pair[1], d))
    return sorted(out)


def keep_set(keys: list[str], dupes: list[tuple[str, str, int]]) -> set[str]:
    """Union-find over duplicate pairs; keep the lexicographically first key of each cluster."""
    parent = {k: k for k in keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _ in dupes:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return {k for k in keys if find(k) == k}
