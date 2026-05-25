"""Context-Mapper (A): strategic-DDD context map producer + pure derivation."""
from core.context_mapper.derive import derive_allowed_dependencies
from core.context_mapper.mapper import run_context_mapper
from core.context_mapper.errors import ContextMapperError

__all__ = ["derive_allowed_dependencies", "run_context_mapper", "ContextMapperError"]
