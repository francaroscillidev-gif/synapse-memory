import re
from typing import List, Set, Tuple

TECH_KEYWORDS = {
    "python", "typescript", "javascript", "rust", "golang", "go", "c++", "c#",
    "docker", "kubernetes", "k8s", "postgres", "postgresql", "sqlite", "mysql",
    "redis", "mongodb", "fastapi", "react", "vue", "angular", "next.js", "tailwind",
    "git", "github", "graphql", "rest", "linux", "mac", "windows", "aws", "gcp"
}

CATEGORY_INDICATORS = {
    "preference": ["preferisco", "prefer", "preferirei", "odio", "amo", "mi piace", "voglio sempre", "evita", "never", "always"],
    "architecture": ["database", "architettura", "microservice", "deploy", "server", "cluster", "produzione", "infra", "backend", "frontend"],
    "rule": ["regola", "obbligatorio", "standard", "convenzione", "policy", "must", "strictly", "guideline"],
    "personal": ["gatto", "cane", "compleanno", "abito a", "famiglia", "chiamo", "hobby", "sport", "amico"],
}


class SmartExtractor:
    @staticmethod
    def extract_tags(text: str) -> List[str]:
        """Extract explicit hashtags and known technology/entity keywords."""
        tags: Set[str] = set()

        # 1. Explicit hashtags like #python #database
        hashtags = re.findall(r"#(\w+)", text)
        for h in hashtags:
            tags.add(h.lower())

        # 2. Tech / entity keywords
        text_lower = text.lower()
        words = set(re.findall(r"\b[a-z0-9\.\+#\-]+\b", text_lower))
        for tech in TECH_KEYWORDS:
            if tech in words:
                tags.add(tech)

        return sorted(list(tags))

    @staticmethod
    def infer_category(text: str, default: str = "general") -> str:
        """Infer the most likely category based on keyword heuristic signals."""
        text_lower = text.lower()
        for cat, indicators in CATEGORY_INDICATORS.items():
            for ind in indicators:
                if ind in text_lower:
                    return cat
        return default
