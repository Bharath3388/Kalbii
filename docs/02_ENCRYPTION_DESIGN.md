# 02 — Custom Encryption Pipeline

> Mandatory rule: **must NOT rely only on AES/Fernet**. We design our own
> reversible cipher composed of well-understood primitives, and we *prove*
> reversibility with round-trip tests.

## 1. Threat model (scope-appropriate)

* In-transit confusion of payload (text + image bytes) so a passive observer
  cannot read content even if they intercept the JSON.
* **Not** intended to replace TLS or production-grade AEAD. Document this
  honestly in the README.

## 2. Pipeline (ENCRYPT direction)

```
plaintext bytes
   │
   ▼  (1) PBKDF2-HMAC-SHA256(passphrase, salt, 200_000) → 32-byte master key
   │      + per-message nonce (16 random bytes)
   │
   ▼  (2) Derive keystream  K = SHA256(master_key || nonce || counter_i)
   │      repeated to len(plaintext)
   │
   ▼  (3) XOR plaintext with keystream     → c1
   │
   ▼  (4) Byte-rotate each byte of c1 left by (master_key[i mod 32] mod 8)  → c2
   │       (reversible bitwise rotation, "shifting" requirement)
   │
   ▼  (5) Block permutation: split c2 into 16-byte blocks; permute byte
   │      indices using a key-seeded Fisher-Yates permutation table π     → c3
   │       (the "scrambling" requirement)
   │
   ▼  (6) Prepend header  =  MAGIC(4) | version(1) | salt(16) | nonce(16)
   │
   ▼  (7) HMAC-SHA256(master_key, header || c3)  → tag(32)
   │
   ▼  (8) Base64-urlsafe encode (header || c3 || tag)
   │
   ▼ ciphertext (ASCII string)
```

DECRYPT is exactly the inverse: base64-decode → split header/body/tag →
verify HMAC (constant-time) → inverse permutation → inverse rotation → XOR
with regenerated keystream.

## 3. Why this satisfies every sub-requirement

| Requirement | Where covered |
|-------------|---------------|
| Custom logic (not pure AES/Fernet) | All of steps 2–5 are hand-rolled |
| Transformation: XOR | Step 3 |
| Transformation: shifting | Step 4 (bit-rotate) |
| Transformation: scrambling | Step 5 (permutation) |
| Encoding: Base64 | Step 8 |
| Reversible pipeline | Each op has a closed-form inverse; round-trip tests |
| Encrypt + Decrypt deliverables | `CustomCipher.encrypt` / `.decrypt` |
| Works for **text** and **image** | Operates on raw `bytes`; helpers wrap str↔bytes and image bytes |

> Note: HMAC + PBKDF2 are used as *integrity / KDF primitives*, not as the
> confidentiality transform. This is acceptable and a good practice; the
> rule forbids relying *only* on AES/Fernet for the cipher itself.

## 4. Reference Python skeleton

```python
# app/crypto/cipher.py
import hmac, hashlib, os, base64, struct
from dataclasses import dataclass

MAGIC = b"KMI1"   # "Kalbii Multi-modal Intelligence v1"

def _pbkdf2(passphrase: bytes, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", passphrase, salt, 200_000, dklen=32)

def _keystream(master: bytes, nonce: bytes, n: int) -> bytes:
    out, counter = bytearray(), 0
    while len(out) < n:
        out += hashlib.sha256(master + nonce + struct.pack(">I", counter)).digest()
        counter += 1
    return bytes(out[:n])

def _rotl8(b: int, r: int) -> int:
    r &= 7
    return ((b << r) | (b >> (8 - r))) & 0xFF if r else b

def _rotr8(b: int, r: int) -> int:
    r &= 7
    return ((b >> r) | (b << (8 - r))) & 0xFF if r else b

def _perm_table(master: bytes, block: int = 16) -> list[int]:
    # deterministic Fisher-Yates seeded by master key
    import random
    rng = random.Random(int.from_bytes(hashlib.sha256(master).digest()[:8], "big"))
    idx = list(range(block))
    rng.shuffle(idx)
    return idx

def _apply_perm(buf: bytes, perm: list[int]) -> bytes:
    block = len(perm)
    out = bytearray(len(buf))
    for i in range(0, len(buf), block):
        chunk = buf[i:i+block]
        if len(chunk) < block:                # tail: leave as-is, length recorded in header? -> we pad
            chunk = chunk + bytes(block - len(chunk))
        for j, p in enumerate(perm):
            out[i + j] = chunk[p]
    return bytes(out)

def _invert_perm(buf: bytes, perm: list[int]) -> bytes:
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return _apply_perm(buf, inv)

@dataclass
class CustomCipher:
    passphrase: bytes

    def encrypt(self, plaintext: bytes) -> str:
        salt  = os.urandom(16)
        nonce = os.urandom(16)
        master = _pbkdf2(self.passphrase, salt)

        # pad to 16-byte boundary, store original length in header
        pad = (-len(plaintext)) % 16
        padded = plaintext + bytes([pad]) * pad
        ks = _keystream(master, nonce, len(padded))

        c1 = bytes(p ^ k for p, k in zip(padded, ks))                       # XOR
        c2 = bytes(_rotl8(b, master[i % 32]) for i, b in enumerate(c1))     # shift
        perm = _perm_table(master)
        c3 = _apply_perm(c2, perm)                                          # scramble

        length = struct.pack(">I", len(plaintext))
        header = MAGIC + b"\x01" + salt + nonce + length                    # 4+1+16+16+4 = 41
        tag = hmac.new(master, header + c3, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(header + c3 + tag).decode()

    def decrypt(self, token: str) -> bytes:
        raw = base64.urlsafe_b64decode(token.encode())
        header, body_tag = raw[:41], raw[41:]
        if header[:4] != MAGIC: raise ValueError("bad magic")
        salt, nonce, length = header[5:21], header[21:37], struct.unpack(">I", header[37:41])[0]
        body, tag = body_tag[:-32], body_tag[-32:]

        master = _pbkdf2(self.passphrase, salt)
        if not hmac.compare_digest(tag, hmac.new(master, header + body, hashlib.sha256).digest()):
            raise ValueError("HMAC mismatch (tampered or wrong key)")

        perm = _perm_table(master)
        c2 = _invert_perm(body, perm)
        c1 = bytes(_rotr8(b, master[i % 32]) for i, b in enumerate(c2))
        ks = _keystream(master, nonce, len(c1))
        padded = bytes(c ^ k for c, k in zip(c1, ks))
        return padded[:length]

    # convenience
    def encrypt_text(self, s: str) -> str:           return self.encrypt(s.encode("utf-8"))
    def decrypt_text(self, t: str) -> str:           return self.decrypt(t).decode("utf-8")
    def encrypt_image_b64(self, img_bytes: bytes) -> str: return self.encrypt(img_bytes)
    def decrypt_image_b64(self, t: str) -> bytes:    return self.decrypt(t)
```

## 5. Mandatory tests

```python
# tests/test_cipher.py
from app.crypto.cipher import CustomCipher

def test_text_round_trip():
    c = CustomCipher(b"super-secret-passphrase")
    msg = "Hairline crack near weld joint — urgent!"
    assert c.decrypt_text(c.encrypt_text(msg)) == msg

def test_image_round_trip(tmp_path):
    c = CustomCipher(b"super-secret-passphrase")
    data = (tmp_path / "x.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 4096) or (tmp_path / "x.png").read_bytes()
    assert c.decrypt(c.encrypt(data)) == data

def test_tamper_detected():
    c = CustomCipher(b"k")
    tok = list(c.encrypt(b"hi"))
    tok[10] = "A" if tok[10] != "A" else "B"
    import pytest
    with pytest.raises(ValueError):
        c.decrypt("".join(tok))
```

## 6. Operational notes

* Passphrase comes from `KMI_PASSPHRASE` env var (never logged).
* Rotate by re-encrypting historical records during a maintenance window.
* For very large images, stream chunks of 1 MB through the cipher (same key,
  fresh nonce per chunk) to bound memory.
