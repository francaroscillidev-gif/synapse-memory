import base64
import os
from typing import Optional
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

SALT_PREFIX = b"synapse_salt_v1_"


class CryptoManager:
    """Provides AES-256-GCM authenticated encryption at rest for sensitive local memories."""

    def __init__(self, passphrase: Optional[str] = None):
        self.passphrase = passphrase or os.environ.get("SYNAPSE_ENCRYPTION_KEY")
        self._aesgcm = None
        if self.passphrase:
            self._aesgcm = self._derive_key(self.passphrase)

    @property
    def is_enabled(self) -> bool:
        return self._aesgcm is not None

    def _derive_key(self, passphrase: str) -> AESGCM:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=SALT_PREFIX,
            iterations=100_000,
        )
        key = kdf.derive(passphrase.encode("utf-8"))
        return AESGCM(key)

    def encrypt(self, plaintext: str) -> str:
        """Encrypts a plaintext string and returns a Base64-encoded payload (nonce + ciphertext)."""
        if not self._aesgcm or not plaintext:
            return plaintext
        nonce = os.urandom(12)  # 96-bit nonce for AES-GCM
        ciphertext = self._aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
        # Store as base64 prefix
        payload = nonce + ciphertext
        return "enc:v1:" + base64.b64encode(payload).decode("ascii")

    def decrypt(self, ciphertext_payload: str) -> str:
        """Decrypts a previously encrypted payload. If unencrypted, returns original string."""
        if not ciphertext_payload.startswith("enc:v1:"):
            return ciphertext_payload
        if not self._aesgcm:
            return "[ENCRYPTED_DATA_KEY_REQUIRED]"

        try:
            raw_b64 = ciphertext_payload[len("enc:v1:"):]
            raw = base64.b64decode(raw_b64)
            nonce = raw[:12]
            ciphertext = raw[12:]
            decrypted = self._aesgcm.decrypt(nonce, ciphertext, None)
            return decrypted.decode("utf-8")
        except Exception:
            return "[DECRYPTION_FAILED_INVALID_KEY]"
