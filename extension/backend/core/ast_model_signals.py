"""
AST Model Signals

Extracts reliable DDD candidate signals from Python AST and enriches the
generated domain model with confidence and traceable sources.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.schemas import DomainModel


class ASTModelSignalExtractor:
	"""Extract AST-based candidates and enrich generated model data."""

	_SKIP_DIR_NAMES = {
		".git",
		".venv",
		"venv",
		"__pycache__",
		"node_modules",
		".mypy_cache",
		".pytest_cache",
		"dist",
		"build",
		"site-packages",
	}

	_SERVICE_SUFFIXES = ("Service", "DomainService", "ApplicationService")
	_ENTITY_EXCLUDED_SUFFIXES = (
		"Service",
		"Repository",
		"Factory",
		"Controller",
		"Manager",
		"Helper",
		"Util",
	)

	def find_python_files(self, workspace_path: str) -> List[str]:
		"""Discover project Python files for AST analysis."""
		root = Path(workspace_path)
		if not root.exists() or not root.is_dir():
			return []

		files: List[str] = []
		for current_root, dirs, filenames in os.walk(root):
			dirs[:] = [d for d in dirs if d not in self._SKIP_DIR_NAMES]

			if "extension/backend" in current_root.replace("\\", "/"):
				continue

			for name in filenames:
				if name.endswith(".py"):
					files.append(str(Path(current_root) / name))

		return files

	def extract_candidates(
		self,
		python_files: List[str],
		grounding_docs: Optional[List[Dict[str, Any]]] = None,
	) -> Dict[str, List[Dict[str, Any]]]:
		"""Extract entity/value object/service/aggregate candidates from AST."""
		result: Dict[str, List[Dict[str, Any]]] = {
			"entities": [],
			"value_objects": [],
			"services": [],
			"aggregates": [],
		}

		for file_path in python_files:
			candidate_sets = self._extract_from_file(file_path)
			for key in result:
				result[key].extend(candidate_sets[key])

		self._deduplicate_by_name(result)

		if grounding_docs:
			for key in result:
				for item in result[key]:
					self._apply_grounding(item, grounding_docs)

		return result

	def enrich_domain_model(
		self,
		model: DomainModel,
		workspace_path: str,
		srs_docs: Optional[List[Dict[str, Any]]] = None,
	) -> DomainModel:
		"""Merge AST candidates into model with confidence and traceability."""
		model_data = model.model_dump(mode="json")

		python_files = self.find_python_files(workspace_path)
		candidates = self.extract_candidates(python_files, srs_docs)
		forbidden_terms = self._build_forbidden_terms(model_data)
		for key in candidates:
			candidates[key] = [
				item
				for item in candidates[key]
				if item.get("name", "").lower() not in forbidden_terms
			]

		bounded_contexts = model_data.get("bounded_contexts", [])
		if not bounded_contexts:
			bounded_contexts = [
				{
					"context_name": "CoreDomain",
					"description": "Auto-created context from AST signals",
					"allowed_dependencies": [],
					"ubiquitous_language": {
						"entities": [],
						"value_objects": [],
						"services": [],
						"aggregates": [],
						"domain_events": [],
					},
				}
			]
			model_data["bounded_contexts"] = bounded_contexts

		for context in bounded_contexts:
			self._normalize_ul_lists(context)

		for key, rule_name in (
			("entities", "Entity"),
			("value_objects", "ValueObject"),
			("services", "Service"),
			("aggregates", "Aggregate"),
		):
			for cand in candidates[key]:
				idx = self._select_context_for_candidate(cand, bounded_contexts)
				if idx is None:
					continue
				target_ul = bounded_contexts[idx].setdefault("ubiquitous_language", {})
				self._normalize_ul_lists(bounded_contexts[idx])
				self._merge_candidates(target_ul[key], [cand], rule_name)

		self._ensure_traceability(model_data, srs_docs)
		return DomainModel(**model_data)

	def _normalize_ul_lists(self, context: Dict[str, Any]) -> None:
		"""Normalize optional ubiquitous_language lists from null to []."""
		ul = context.setdefault("ubiquitous_language", {})
		for key in ("entities", "value_objects", "services", "aggregates", "domain_events"):
			if ul.get(key) is None:
				ul[key] = []

	def _build_forbidden_terms(self, model_data: Dict[str, Any]) -> set[str]:
		"""Collect terms that should not be added as canonical model items."""
		forbidden = set()
		global_rules = model_data.get("global_rules") or {}
		for term in global_rules.get("banned_global_terms", []) or []:
			forbidden.add(str(term).strip().lower())

		for ctx in model_data.get("bounded_contexts", []) or []:
			ul = (ctx or {}).get("ubiquitous_language", {})
			for entity in ul.get("entities", []) or []:
				for syn in (entity or {}).get("synonyms_to_avoid", []) or []:
					forbidden.add(str(syn).strip().lower())

		return {t for t in forbidden if t}

	def _select_context_for_candidate(
		self,
		candidate: Dict[str, Any],
		bounded_contexts: List[Dict[str, Any]],
	) -> Optional[int]:
		"""Pick best bounded context; return None if no reasonable match."""
		if len(bounded_contexts) == 1:
			return 0

		name = candidate.get("name", "")
		if not name:
			return None

		name_l = name.lower()
		best_idx = None
		best_score = 0

		for idx, ctx in enumerate(bounded_contexts):
			ul = (ctx or {}).get("ubiquitous_language", {})
			score = 0

			for bucket in ("entities", "value_objects", "services", "aggregates"):
				for item in ul.get(bucket, []) or []:
					item_name = str((item or {}).get("name", "")).lower()
					if not item_name:
						continue
					if item_name == name_l:
						score += 10
					elif item_name in name_l or name_l in item_name:
						score += 4

			ctx_name = str((ctx or {}).get("context_name", "")).lower()
			if ctx_name and any(token in name_l for token in re.split(r"(?<!^)(?=[A-Z])", ctx.get("context_name", "")) if token):
				score += 1

			if score > best_score:
				best_score = score
				best_idx = idx

		# Avoid dumping unmatched candidates into first context.
		if best_score <= 0:
			return None
		return best_idx

	def _extract_from_file(self, file_path: str) -> Dict[str, List[Dict[str, Any]]]:
		out = {"entities": [], "value_objects": [], "services": [], "aggregates": []}

		try:
			source = Path(file_path).read_text(encoding="utf-8")
			tree = ast.parse(source)
		except Exception:
			return out

		for node in tree.body:
			if isinstance(node, ast.ClassDef):
				class_info = self._class_info(node)
				source_ref = {
					"file": file_path,
					"line": node.lineno,
					"rule": "AST_CLASS_DEF",
					"evidence": f"class {node.name}",
				}

				if self._is_entity_candidate(class_info):
					confidence = self._score_entity(class_info)
					out["entities"].append(
						{
							"name": node.name,
							"description": "AST-discovered entity candidate",
							"confidence": confidence,
							"synonyms_to_avoid": [],
							"sources": [source_ref],
						}
					)

				if self._is_value_object_candidate(class_info):
					confidence = self._score_value_object(class_info)
					out["value_objects"].append(
						{
							"name": node.name,
							"attributes": sorted(class_info["attributes"]),
							"description": "AST-discovered value object candidate",
							"confidence": confidence,
							"sources": [source_ref],
						}
					)

				if self._is_service_candidate(class_info):
					out["services"].append(
						{
							"name": node.name,
							"description": "AST-discovered service candidate",
							"confidence": 0.75,
							"sources": [source_ref],
						}
					)

				if self._is_aggregate_candidate(class_info):
					out["aggregates"].append(
						{
							"name": node.name,
							"description": "AST-discovered aggregate candidate",
							"confidence": 0.72,
							"sources": [source_ref],
						}
					)

		return out

	def _class_info(self, node: ast.ClassDef) -> Dict[str, Any]:
		methods: List[str] = []
		attributes: List[str] = []
		decorators: List[str] = []

		for dec in node.decorator_list:
			if isinstance(dec, ast.Name):
				decorators.append(dec.id)
			elif isinstance(dec, ast.Attribute):
				decorators.append(dec.attr)

		for item in node.body:
			if isinstance(item, ast.FunctionDef):
				methods.append(item.name)

				for inner in ast.walk(item):
					if isinstance(inner, ast.Assign):
						for target in inner.targets:
							attr = self._extract_self_attribute(target)
							if attr:
								attributes.append(attr)
					elif isinstance(inner, ast.AnnAssign):
						attr = self._extract_self_attribute(inner.target)
						if attr:
							attributes.append(attr)

		return {
			"name": node.name,
			"methods": methods,
			"attributes": set(attributes),
			"decorators": decorators,
		}

	def _extract_self_attribute(self, target: ast.AST) -> Optional[str]:
		if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
			if target.value.id == "self":
				return target.attr
		return None

	def _is_entity_candidate(self, info: Dict[str, Any]) -> bool:
		name = info["name"]
		if any(name.endswith(sfx) for sfx in self._ENTITY_EXCLUDED_SUFFIXES):
			return False
		if not name or not name[0].isupper() or "_" in name:
			return False

		method_count = len([m for m in info["methods"] if not m.startswith("__")])
		attrs_count = len(info["attributes"])
		return "__init__" in info["methods"] and (attrs_count >= 2 or method_count >= 2)

	def _is_value_object_candidate(self, info: Dict[str, Any]) -> bool:
		methods = set(info["methods"])
		has_dataclass = "dataclass" in info["decorators"]
		has_value_name = any(tag in info["name"] for tag in ("Value", "VO", "Money", "Address"))
		has_immutable_signals = "__eq__" in methods or "__hash__" in methods
		mutator_exists = any(
			m.startswith(("set", "update", "add", "remove"))
			for m in methods
			if not m.startswith("__")
		)
		return (has_dataclass or has_value_name or has_immutable_signals) and not mutator_exists

	def _is_service_candidate(self, info: Dict[str, Any]) -> bool:
		name = info["name"]
		if name.endswith(self._SERVICE_SUFFIXES):
			return True
		non_dunder = [m for m in info["methods"] if not m.startswith("__")]
		return len(non_dunder) >= 2 and len(info["attributes"]) == 0

	def _is_aggregate_candidate(self, info: Dict[str, Any]) -> bool:
		name = info["name"]
		if name.endswith("Aggregate"):
			return True
		attrs = {a.lower() for a in info["attributes"]}
		methods = [m.lower() for m in info["methods"]]
		has_collection = any(tag in attrs for tag in ("items", "children", "lines", "entries"))
		has_guard_methods = any(m.startswith(("add", "remove", "apply")) for m in methods)
		return has_collection and has_guard_methods

	def _score_entity(self, info: Dict[str, Any]) -> float:
		score = 0.55
		if len(info["attributes"]) >= 2:
			score += 0.15
		if len([m for m in info["methods"] if not m.startswith("__")]) >= 2:
			score += 0.15
		return min(round(score, 2), 0.95)

	def _score_value_object(self, info: Dict[str, Any]) -> float:
		score = 0.58
		if "dataclass" in info["decorators"]:
			score += 0.15
		if "__eq__" in info["methods"]:
			score += 0.12
		return min(round(score, 2), 0.92)

	def _deduplicate_by_name(self, candidate_map: Dict[str, List[Dict[str, Any]]]) -> None:
		for key, items in candidate_map.items():
			by_name: Dict[str, Dict[str, Any]] = {}
			for item in items:
				name = item.get("name", "")
				if not name:
					continue
				existing = by_name.get(name.lower())
				if not existing or item.get("confidence", 0) > existing.get("confidence", 0):
					by_name[name.lower()] = item
				elif existing:
					existing_sources = existing.setdefault("sources", [])
					existing_sources.extend(item.get("sources", []))
			candidate_map[key] = list(by_name.values())

	def _apply_grounding(self, item: Dict[str, Any], docs: List[Dict[str, Any]]) -> None:
		name = item.get("name", "")
		if not name:
			return

		best = self._find_grounding(name, docs)
		if not best:
			return

		file_path, line_num, snippet = best
		item["confidence"] = min(round(float(item.get("confidence", 0.5)) + 0.1, 2), 0.99)
		item.setdefault("sources", []).append(
			{
				"file": file_path,
				"line": line_num,
				"rule": "SRS_GROUNDING",
				"evidence": snippet[:140],
			}
		)

	def _find_grounding(
		self,
		term: str,
		docs: List[Dict[str, Any]],
	) -> Optional[Tuple[str, int, str]]:
		needle = re.compile(rf"\b{re.escape(term)}\b", flags=re.IGNORECASE)
		for doc in docs:
			text = doc.get("content", "")
			if not text:
				continue
			for idx, line in enumerate(text.splitlines(), start=1):
				line_stripped = line.strip()
				# Skip negative synonym constraints as grounding evidence.
				if "do not use" in line_stripped.lower():
					continue
				if needle.search(line_stripped):
					return (doc.get("path", "srs_document"), idx, line_stripped)
		return None

	def _merge_candidates(
		self,
		existing: List[Dict[str, Any]],
		candidates: List[Dict[str, Any]],
		rule_name: str,
	) -> None:
		existing_by_name = {item.get("name", "").lower(): item for item in existing}
		for cand in candidates:
			key = cand.get("name", "").lower()
			if not key:
				continue

			if key not in existing_by_name:
				existing.append(cand)
				existing_by_name[key] = cand
			else:
				current = existing_by_name[key]
				current["confidence"] = max(
					float(current.get("confidence", 0.5)),
					float(cand.get("confidence", 0.5)),
				)
				current.setdefault("sources", [])
				current["sources"].extend(cand.get("sources", []))
				if not current.get("description"):
					current["description"] = cand.get("description", f"{rule_name} inference")

	def _ensure_traceability(
		self,
		model_data: Dict[str, Any],
		srs_docs: Optional[List[Dict[str, Any]]],
	) -> None:
		if not srs_docs:
			srs_docs = []

		for ctx in model_data.get("bounded_contexts", []):
			ul = ctx.get("ubiquitous_language", {})
			for key in ("entities", "value_objects", "services", "aggregates"):
				for item in ul.get(key, []) or []:
					if item.get("confidence") is None:
						item["confidence"] = 0.5
					item["confidence"] = round(min(max(float(item["confidence"]), 0.0), 1.0), 2)

					item.setdefault("sources", [])
					if not item["sources"]:
						grounded = self._find_grounding(item.get("name", ""), srs_docs)
						if grounded:
							file_path, line_num, snippet = grounded
							item["sources"].append(
								{
									"file": file_path,
									"line": line_num,
									"rule": "SRS_ENTITY_MATCH",
									"evidence": snippet[:140],
								}
							)
						else:
							item["sources"].append(
								{
									"file": "generated",
									"line": 1,
									"rule": "LLM_SYNTHESIS",
									"evidence": "Generated by synthesis stage",
								}
							)
