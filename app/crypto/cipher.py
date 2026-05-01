"""Custom reversible cipher.

Pipeline (encrypt direction):
    plaintext bytes
        -> PBKDF2-HMAC-SHA256 derived 32-byte master key (per-message salt)
        -> XOR with SHA256-counter keystream  (transformation)
        -> per-byte left bit-rotation by master_key[i % 32] mod 8 (shifting)
        -> 16-byte block permutation seeded by master key (scrambling)
        -> prepend header (magic | version | salt | nonce | length)
        -> append HMAC-SHA256 tag (integrity)
        -> Base64-urlsafe encode

Decrypt is the exact inverse with constant-time HMAC verification.
"""

from __future__ import annotations
import base64
import hashlib
import hmac
import os
import random
import struct
from dataclasses import dataclass
from typing import List

MAGIC = b"KMI1"
VERSION = b"\x01"
HEADER_LEN = 4 + 1 + 16 + 16 + 4   # 41 bytes
BLOCK = 16
TAG_LEN = 32


def _pbkdf2(passphrase: bytes, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", passphrase, salt, 200_000, dklen=32)


def _keystream(master: bytes, nonce: bytes, n: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < n:
        out += hashlib.sha256(master + nonce + struct.pack(">I", counter)).digest()
        counter += 1
    return bytes(out[:n])


def _rotl8(b: int, r: int) -> int:
    r &= 7
    if r == 0:
        return b
    return ((b << r) | (b >> (8 - r))) & 0xFF


def _rotr8(b: int, r: int) -> int:
    r &= 7
    if r == 0:
        return b
    return ((b >> r) | (b << (8 - r))) & 0xFF


def _perm_table(master: bytes) -> List[int]:
    seed = int.from_bytes(hashlib.sha256(b"perm:" + master).digest()[:8], "big")
    rng = random.Random(seed)
    idx = list(range(BLOCK))
    rng.shuffle(idx)
    return idx


def _apply_perm(buf: bytes, perm: List[int]) -> bytes:
    assert len(buf) % BLOCK == 0
    out = bytearray(len(buf))
    for i in range(0, len(buf), BLOCK):
        chunk = buf[i:i + BLOCK]
        for j, p in enumerate(perm):
            out[i + j] = chunk[p]
    return bytes(out)


def _invert_perm(buf: bytes, perm: List[int]) -> bytes:
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return _apply_perm(buf, inv)


@dataclass
class CustomCipher:
    passphrase: bytes

    @classmethod
    def from_str(cls, s: str) -> "CustomCipher":
        return cls(s.encode("utf-8"))

    def encrypt(self, plaintext: bytes) -> str:
        if not isinstance(plaintext, (bytes, bytearray)):
            raise TypeError("encrypt expects bytes")
        salt = os.urandom(16)
        nonce = os.urandom(16)
        master = _pbkdf2(self.passphrase, salt)

        pad = (-len(plaintext)) % BLOCK
        padded = bytes(plaintext) + (b"\x00" * pad)
        ks = _keystream(master, nonce, len(padded))

        c1 = bytes(p ^ k for p, k in zip(padded, ks))
        c2 = bytes(_rotl8(b, master[i % 32]) for i, b in enumerate(c1))
        perm = _perm_table(master)
        c3 = _apply_perm(c2, perm)

        length = struct.pack(">I", len(plaintext))
        header = MAGIC + VERSION + salt + nonce + length
        tag = hmac.new(master, header + c3, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(header + c3 + tag).decode("ascii")

    def decrypt(self, token: str) -> bytes:
        try:
            raw = base64.urlsafe_b64decode(token.encode("ascii"))
        except Exception as e:                                          # noqa: BLE001
            raise ValueError(f"invalid base64: {e}") from e

        if len(raw) < HEADER_LEN + TAG_LEN:
            raise ValueError("ciphertext too short")
        header = raw[:HEADER_LEN]
        body = raw[HEADER_LEN:-TAG_LEN]
        tag = raw[-TAG_LEN:]

        if header[:4] != MAGIC:
            raise ValueError("bad magic")
        if header[4:5] != VERSION:
            raise ValueError("unsupported version")
        salt = header[5:21]
        nonce = header[21:37]
        length = struct.unpack(">I", header[37:41])[0]

        master = _pbkdf2(self.passphrase, salt)
        expected = hmac.new(master, header + body, hashlib.sha256).digest()
        if not hmac.compare_digest(tag, expected):
            raise ValueError("HMAC mismatch (tampered or wrong key)")

        if len(body) % BLOCK != 0:
            raise ValueError("body not block-aligned")

        perm = _perm_table(master)
        c2 = _invert_perm(body, perm)
        c1 = bytes(_rotr8(b, master[i % 32]) for i, b in enumerate(c2))
        ks = _keystream(master, nonce, len(c1))
        padded = bytes(c ^ k for c, k in zip(c1, ks))
        if length > len(padded):
            raise ValueError("declared length exceeds payload")
        return padded[:length]

    # --- convenience ---------------------------------------------------
    def encrypt_text(self, s: str) -> str:
        return self.encrypt(s.encode("utf-8"))

    def decrypt_text(self, token: str) -> str:
        return self.decrypt(token).decode("utf-8")
