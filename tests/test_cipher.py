import os
import pytest

from app.crypto.cipher import CustomCipher


def test_text_round_trip():
    c = CustomCipher.from_str("a-secret")
    msg = "Hairline crack near weld joint — urgent!"
    assert c.decrypt_text(c.encrypt_text(msg)) == msg


def test_empty_round_trip():
    c = CustomCipher.from_str("a-secret")
    assert c.decrypt(c.encrypt(b"")) == b""


def test_binary_round_trip():
    c = CustomCipher.from_str("a-secret")
    data = bytes(range(256)) * 32                      # 8192 bytes, all values
    assert c.decrypt(c.encrypt(data)) == data


def test_image_like_bytes_round_trip():
    c = CustomCipher.from_str("a-secret")
    data = b"\x89PNG\r\n\x1a\n" + os.urandom(50_000)
    assert c.decrypt(c.encrypt(data)) == data


def test_tamper_detected():
    c = CustomCipher.from_str("a-secret")
    tok = list(c.encrypt(b"hello world"))
    # flip a character in the body
    i = 60
    tok[i] = "A" if tok[i] != "A" else "B"
    with pytest.raises(ValueError):
        c.decrypt("".join(tok))


def test_wrong_key_fails():
    a = CustomCipher.from_str("key-A")
    b = CustomCipher.from_str("key-B")
    tok = a.encrypt(b"secret payload")
    with pytest.raises(ValueError):
        b.decrypt(tok)


def test_distinct_ciphertexts_for_same_plaintext():
    c = CustomCipher.from_str("a-secret")
    t1 = c.encrypt(b"same")
    t2 = c.encrypt(b"same")
    assert t1 != t2                                    # nonce ensures uniqueness
    assert c.decrypt(t1) == c.decrypt(t2) == b"same"
