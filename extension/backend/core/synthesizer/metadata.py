"""Mechanical metadata + global rules defaults."""

import time
from core.schemas import ProjectMetadata, GlobalRules


def build_default_metadata() -> ProjectMetadata:
    return ProjectMetadata(
        version="1.0",
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        description="Domain model generated from SRS via DDD-Enforcer pipeline",
    )


def build_default_global_rules() -> GlobalRules:
    return GlobalRules(
        naming_convention="PascalCase",
        banned_global_terms=["Manager", "Util", "Helper", "Data", "Info"],
    )
