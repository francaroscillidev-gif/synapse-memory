import re
from typing import Tuple

# Patterns for secrets and PII
SECRET_PATTERNS = [
    # GitHub Tokens
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{36,255}\b"), "[REDACTED_GITHUB_TOKEN]"),
    # OpenAI & generic API keys
    (re.compile(r"\bsk-[a-zA-Z0-9]{20,T3BlbkFJ[a-zA-Z0-9]{20,}\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"\bsk-proj-[a-zA-Z0-9_\-]{40,}\b"), "[REDACTED_API_KEY]"),
    # Anthropic API Keys
    (re.compile(r"\bsk-ant-[a-zA-Z0-9_\-]{40,}\b"), "[REDACTED_ANTHROPIC_KEY]"),
    # JWT Tokens
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"), "[REDACTED_JWT]"),
    # AWS Access Key ID
    (re.compile(r"\b(AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16}\b"), "[REDACTED_AWS_KEY]"),
    # Private Key Headers
    (re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----[\s\S]*?-----END [A-Z ]+ PRIVATE KEY-----"), "[REDACTED_PRIVATE_KEY]"),
    # Generic password assignments (e.g. password=xyz, pwd: xyz)
    (re.compile(r"(?i)\b(password|passwd|pwd|secret)\s*[:=]\s*['\"]?([^\s'\"]{6,})['\"]?"), r"\1=[REDACTED_SECRET]"),
    # Credit Card Numbers
    (re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b"), "[REDACTED_CREDIT_CARD]"),
]


class SecurityRedactor:
    """Sanitizes text by detecting and redacting sensitive credentials and PII."""

    @staticmethod
    def redact(text: str) -> Tuple[str, int]:
        """
        Redacts credentials and secrets from text.
        Returns: (sanitized_text, count_of_redactions)
        """
        redacted_text = text
        total_redactions = 0

        for pattern, replacement in SECRET_PATTERNS:
            matches = list(pattern.finditer(redacted_text))
            if matches:
                total_redactions += len(matches)
                redacted_text = pattern.sub(replacement, redacted_text)

        return redacted_text, total_redactions
