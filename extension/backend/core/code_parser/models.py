from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class ParameterRecord:
    name: str
    annotation: Optional[str] = None
    kind: str = "positional_or_keyword"
    has_default: bool = False
    default: Optional[str] = None

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.annotation,
            "annotation": self.annotation,
            "kind": self.kind,
            "has_default": self.has_default,
            "default": self.default,
        }


@dataclass(slots=True)
class ClassRecord:
    name: str
    line: int
    bases: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    base_names: List[str] = field(default_factory=list)
    class_attributes: List[str] = field(default_factory=list)
    instance_attributes: List[str] = field(default_factory=list)
    mutation_methods: List[str] = field(default_factory=list)
    is_dataclass: bool = False
    is_frozen: bool = False

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "line": self.line,
            "bases": self.bases,
            "methods": self.methods,
            "decorators": self.decorators,
            "base_names": self.base_names,
            "class_attributes": self.class_attributes,
            "instance_attributes": self.instance_attributes,
            "mutation_methods": self.mutation_methods,
            "is_dataclass": self.is_dataclass,
            "is_frozen": self.is_frozen,
        }


@dataclass(slots=True)
class FunctionRecord:
    name: str
    line: int
    parameters: List[ParameterRecord] = field(default_factory=list)
    in_class: Optional[str] = None
    in_function: Optional[str] = None
    nesting_level: int = 0
    return_annotation: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    is_async: bool = False

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "line": self.line,
            "parameters": [parameter.to_public_dict() for parameter in self.parameters],
            "in_class": self.in_class,
            "in_function": self.in_function,
            "nesting_level": self.nesting_level,
            "return_type": self.return_annotation,
            "return_annotation": self.return_annotation,
            "decorators": self.decorators,
            "is_async": self.is_async,
        }


@dataclass(slots=True)
class ImportRecord:
    module: str
    line: int
    import_type: str
    names: List[str] = field(default_factory=list)
    asname: Optional[str] = None
    level: int = 0
    members: List[Dict[str, Optional[str]]] = field(default_factory=list)
    imported_modules: List[str] = field(default_factory=list)

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "module": self.module,
            "normalized_module": self.module,
            "type": self.import_type,
            "line": self.line,
            "names": self.names,
            "asname": self.asname,
            "level": self.level,
            "is_relative": self.level > 0,
            "members": self.members,
            "imported_modules": self.imported_modules,
        }


@dataclass(slots=True)
class AssignmentRecord:
    line: int
    target: Optional[str]
    target_kind: str
    in_class: Optional[str] = None
    in_function: Optional[str] = None
    nesting_level: int = 0
    annotation: Optional[str] = None
    operation: str = "="
    value_type: Optional[str] = None
    value_shape: Optional[str] = None
    value: Optional[str] = None

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "line": self.line,
            "target": self.target,
            "target_kind": self.target_kind,
            "in_class": self.in_class,
            "in_function": self.in_function,
            "nesting_level": self.nesting_level,
            "annotation": self.annotation,
            "operation": self.operation,
            "value_type": self.value_type,
            "value_shape": self.value_shape,
            "value": self.value,
        }


@dataclass(slots=True)
class FunctionCallRecord:
    line: int
    function: Optional[str]
    args: List[str] = field(default_factory=list)
    kwargs: Dict[str, str] = field(default_factory=dict)
    receiver: Optional[str] = None
    call_kind: str = "unknown"
    in_class: Optional[str] = None
    in_function: Optional[str] = None
    nesting_level: int = 0

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "line": self.line,
            "function": self.function,
            "args": self.args,
            "kwargs": self.kwargs,
            "receiver": self.receiver,
            "call_kind": self.call_kind,
            "in_class": self.in_class,
            "in_function": self.in_function,
            "nesting_level": self.nesting_level,
        }


@dataclass(slots=True)
class ParseResult:
    filename: str
    classes: List[ClassRecord] = field(default_factory=list)
    imports: List[ImportRecord] = field(default_factory=list)
    functions: List[FunctionRecord] = field(default_factory=list)
    assignments: List[AssignmentRecord] = field(default_factory=list)
    function_calls: List[FunctionCallRecord] = field(default_factory=list)

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "classes": [record.to_public_dict() for record in self.classes],
            "imports": [record.to_public_dict() for record in self.imports],
            "functions": [record.to_public_dict() for record in self.functions],
            "assignments": [record.to_public_dict() for record in self.assignments],
            "function_calls": [
                record.to_public_dict() for record in self.function_calls
            ],
        }
