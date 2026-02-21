"""
AST Model Signals (Production Ready)

Extracts reliable DDD candidate signals from Python AST and enriches the
generated domain model with confidence and traceable sources.
"""

from __future__ import annotations

import ast
import os
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from core.schemas import DomainModel

# Logging ayarları
logger = logging.getLogger(__name__)

class ASTModelSignalExtractor:
    """
    Extract AST-based candidates and enrich generated model data.
    Refactored for robustness, avoiding domain overfitting.
    """

    # Varsayılan yoksayılacak klasörler
    DEFAULT_SKIP_DIRS = {
        ".git", ".venv", "venv", "__pycache__", "node_modules",
        ".mypy_cache", ".pytest_cache", "dist", "build", "site-packages",
        "migrations", "tests", "test"
    }

    # Servis olarak kabul edilebilecek son ekler (Standart Naming Convention)
    _SERVICE_SUFFIXES = ("Service", "Manager", "Processor", "Handler", "UseCase", "Provider")
    
    # Entity olamayacak sınıflar (False Positive elemek için)
    _ENTITY_EXCLUDED_SUFFIXES = (
        "Service", "Repository", "Factory", "Controller", "Manager",
        "Helper", "Util", "Dto", "DTO", "Exception", "Config", "Settings"
    )

    def __init__(self, ignore_paths: Optional[List[str]] = None):
        """
        Args:
            ignore_paths: Taranmayacak ekstra dosya yolları listesi.
        """
        self.skip_dirs = self.DEFAULT_SKIP_DIRS
        self.ignore_paths = ignore_paths or []

    def find_python_files(self, workspace_path: str) -> List[str]:
        """Discover project Python files for AST analysis robustly."""
        root = Path(workspace_path)
        if not root.exists() or not root.is_dir():
            logger.warning(f"Workspace path not found: {workspace_path}")
            return []

        files: List[str] = []
        for current_root, dirs, filenames in os.walk(root):
            # Gereksiz klasörleri atla
            dirs[:] = [d for d in dirs if d not in self.skip_dirs]

            # Yapılandırılmış ignore path kontrolü
            rel_path = os.path.relpath(current_root, root).replace("\\", "/")
            if any(ignored in rel_path for ignored in self.ignore_paths):
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
            try:
                candidate_sets = self._extract_from_file(file_path)
                for key in result:
                    result[key].extend(candidate_sets[key])
            except Exception as e:
                logger.error(f"Error parsing file {file_path}: {e}")
                continue

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
        # Pydantic modelini dict'e çeviriyoruz
        model_data = model.model_dump(mode="json")

        python_files = self.find_python_files(workspace_path)
        candidates = self.extract_candidates(python_files, srs_docs)
        
        forbidden_terms = self._build_forbidden_terms(model_data)

        # Temizleme işlemi
        for key in candidates:
            candidates[key] = [
                item
                for item in candidates[key]
                if item.get("name", "").lower() not in forbidden_terms
            ]

        # Eğer hiç context yoksa default context oluştur
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

        # Adayları uygun Context'e yerleştir
        for key, rule_name in (
            ("entities", "Entity"),
            ("value_objects", "ValueObject"),
            ("services", "Service"),
            ("aggregates", "Aggregate"),
        ):
            for cand in candidates[key]:
                idx = self._select_context_for_candidate(cand, bounded_contexts)
                if idx is None:
                    # En uygun context bulunamazsa ilki veya "Core" seçilebilir
                    # Şimdilik production safe olması için atlıyoruz veya default'a atabiliriz
                    idx = 0 
                
                target_ul = bounded_contexts[idx].setdefault("ubiquitous_language", {})
                self._normalize_ul_lists(bounded_contexts[idx])
                self._merge_candidates(target_ul[key], [cand], rule_name)

        self._ensure_traceability(model_data, srs_docs)
        return DomainModel(**model_data)

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

                # --- Detection Logic Pipeline ---
                
                # 1. Service Detection (Dependency Injection Check)
                if self._is_service_candidate(class_info):
                    out["services"].append({
                        "name": node.name,
                        "description": "AST-discovered service (business logic/handler)",
                        "confidence": 0.85, # High confidence for services
                        "sources": [source_ref],
                    })
                    continue # Bir sınıf servisse Entity olamaz, devam et.

                # 2. Value Object Detection (Immutability/Structure Check)
                if self._is_value_object_candidate(class_info):
                    out["value_objects"].append({
                        "name": node.name,
                        "attributes": sorted(class_info["attributes"]),
                        "description": "AST-discovered Value Object (immutable structure)",
                        "confidence": 0.70,
                        "sources": [source_ref],
                    })
                    continue

                # 3. Aggregate Detection (Collection Management Check)
                if self._is_aggregate_candidate(class_info):
                    out["aggregates"].append({
                        "name": node.name,
                        "description": "AST-discovered Aggregate Root",
                        "confidence": 0.75,
                        "sources": [source_ref],
                    })
                    # Aggregate aynı zamanda Entity'dir, o yüzden continue demiyoruz.

                # 4. Entity Detection (Identity Check)
                if self._is_entity_candidate(class_info):
                    confidence = self._score_entity(class_info)
                    out["entities"].append({
                        "name": node.name,
                        "description": "AST-discovered Entity",
                        "confidence": confidence,
                        "synonyms_to_avoid": [],
                        "sources": [source_ref],
                    })

        return out

    def _class_info(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Parses a class node to extract methods, attributes, decorators, and base classes."""
        methods: List[str] = []
        attributes: Set[str] = set()
        decorators: List[str] = []
        bases: List[str] = []
        
        # Base classes
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(base.attr)

        # Decorators
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decorators.append(dec.attr)
            elif isinstance(dec, ast.Call):  # @dataclass(frozen=True) gibi durumlar
                if isinstance(dec.func, ast.Name):
                    decorators.append(dec.func.id)

        # Body analysis
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(item.name)
                
                # __init__ içinde self.x = y atamalarını yakala
                if item.name == "__init__":
                    for inner in ast.walk(item):
                        if isinstance(inner, ast.Assign):
                            for target in inner.targets:
                                attr = self._extract_self_attribute(target)
                                if attr: attributes.add(attr)
                        elif isinstance(inner, ast.AnnAssign): # self.x: int = 1
                            attr = self._extract_self_attribute(inner.target)
                            if attr: attributes.add(attr)

            # Sınıf seviyesi attribute'lar (Type hints)
            elif isinstance(item, ast.AnnAssign):
                if isinstance(item.target, ast.Name):
                    attributes.add(item.target.id)

        return {
            "name": node.name,
            "methods": methods,
            "attributes": attributes,
            "decorators": decorators,
            "bases": bases
        }

    def _extract_self_attribute(self, target: ast.AST) -> Optional[str]:
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
            if target.value.id == "self":
                return target.attr
        return None

    def _is_service_candidate(self, info: Dict[str, Any]) -> bool:
        """
        Determines if a class is likely a Domain Service.
        Criteria: Suffix match OR (No State + Action Methods).
        """
        name = info["name"]
        
        # 1. İsimlendirme Konvansiyonu (Güçlü Sinyal)
        if any(name.endswith(sfx) for sfx in self._SERVICE_SUFFIXES):
            return True

        # 2. Davranışsal Analiz (Zayıf Sinyal ama Overfit'i engeller)
        # Eğer attribute'ları varsa bile bunlar 'Repository' veya 'Service' mi? (Dependency Injection)
        # AST ile değişken tiplerini tam anlamak zordur ama isimlere bakabiliriz.
        dependencies = [attr for attr in info["attributes"] 
                        if attr.lower().endswith(('repo', 'repository', 'service', 'client', 'adapter'))]
        
        has_dependencies = len(dependencies) > 0
        has_business_methods = len([m for m in info["methods"] if not m.startswith("__")]) >= 1
        
        # Eğer dependency'si var ve iş metodu varsa servistir.
        return has_dependencies and has_business_methods

    def _is_value_object_candidate(self, info: Dict[str, Any]) -> bool:
        """
        Value Object detection based on immutability and structure.
        Avoids hardcoded names like 'Money'.
        """
        methods = set(info["methods"])
        decorators = info["decorators"]
        bases = info["bases"]

        # 1. Açıkça belirtilmiş mi? (Base Class veya Decorator)
        if "ValueObject" in bases or "frozen" in str(decorators): 
            return True

        # 2. Yapısal Eşitlik (Equality) var mı?
        has_eq = "__eq__" in methods
        
        # 3. Setter'ı var mı? (Varsa VO değildir)
        has_setter = any(m.startswith("set_") or m.startswith("set") for m in methods)
        
        # 4. Veri taşıyıcı mı? (Data Class behavior)
        is_dataclass = "dataclass" in decorators
        
        # VO Kriteri: (Dataclass VE Setter Yok) VEYA (Eşitlik metodu var VE Setter Yok)
        return (is_dataclass and not has_setter) or (has_eq and not has_setter)

    def _is_aggregate_candidate(self, info: Dict[str, Any]) -> bool:
        """
        Detects Aggregate Roots.
        Criteria: Named 'Root', 'Aggregate' OR manages lists of items.
        """
        name = info["name"]
        attributes = info["attributes"]
        
        # 1. İsimlendirme
        if name.endswith("Aggregate") or "Root" in name:
            return True

        # 2. Koleksiyon Yönetimi (Basit bir heuristic)
        # Bir sınıf 'items', 'lines', 'rows' gibi çoğul isimli bir liste yönetiyorsa Aggregate olabilir.
        has_collection = any(attr in ("items", "entries", "lines", "products", "orders") for attr in attributes)
        
        # Aggregate metotları: add_item, remove_item vb.
        has_management_methods = any(m.startswith(("add_", "remove_", "update_")) for m in info["methods"])

        return has_collection and has_management_methods

    def _is_entity_candidate(self, info: Dict[str, Any]) -> bool:
        """
        Standard Entity check.
        Criteria: Not excluded, has ID, has lifecycle methods.
        """
        name = info["name"]
        
        # Yasaklı suffix varsa entity değildir.
        if any(name.endswith(sfx) for sfx in self._ENTITY_EXCLUDED_SUFFIXES):
            return False
            
        if not name or not name[0].isupper() or "_" in name:
            return False

        # Kimlik Kontrolü (Identity Pattern) - En önemli Entity özelliği
        attributes = info["attributes"]
        has_id = any(attr.lower() in ("id", "uuid", "guid", "code", "identifier") or attr.endswith("_id") for attr in attributes)
        
        # Entity olması için ID'si olmalı VEYA metotları ve attribute'ları zengin olmalı
        method_count = len([m for m in info["methods"] if not m.startswith("__")])
        attrs_count = len(attributes)
        
        return has_id or (attrs_count >= 2 and method_count >= 1)

    def _score_entity(self, info: Dict[str, Any]) -> float:
        """Calculates a confidence score for an entity candidate."""
        score = 0.50
        
        # ID'si varsa kesinlikle entity olma ihtimali artar
        if any(a.lower() == "id" or a.endswith("_id") for a in info["attributes"]):
            score += 0.30
            
        if len(info["attributes"]) >= 2:
            score += 0.10
            
        if len([m for m in info["methods"] if not m.startswith("__")]) >= 2:
            score += 0.05
            
        return min(round(score, 2), 0.99)

    # --- Utility Methods (Değişmedi, aynı mantık) ---
    
    def _normalize_ul_lists(self, context: Dict[str, Any]) -> None:
        ul = context.setdefault("ubiquitous_language", {})
        for key in ("entities", "value_objects", "services", "aggregates", "domain_events"):
            if ul.get(key) is None:
                ul[key] = []

    def _build_forbidden_terms(self, model_data: Dict[str, Any]) -> Set[str]:
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
        if len(bounded_contexts) == 1:
            return 0
        name = candidate.get("name", "")
        if not name: return None
        name_l = name.lower()
        best_idx = None
        best_score = 0
        for idx, ctx in enumerate(bounded_contexts):
            ul = (ctx or {}).get("ubiquitous_language", {})
            score = 0
            for bucket in ("entities", "value_objects", "services", "aggregates"):
                for item in ul.get(bucket, []) or []:
                    item_name = str((item or {}).get("name", "")).lower()
                    if not item_name: continue
                    if item_name == name_l: score += 10
                    elif item_name in name_l or name_l in item_name: score += 4
            ctx_name = str((ctx or {}).get("context_name", "")).lower()
            if ctx_name and ctx_name in name_l: score += 2
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def _deduplicate_by_name(self, candidate_map: Dict[str, List[Dict[str, Any]]]) -> None:
        for key, items in candidate_map.items():
            by_name: Dict[str, Dict[str, Any]] = {}
            for item in items:
                name = item.get("name", "")
                if not name: continue
                existing = by_name.get(name.lower())
                if not existing or item.get("confidence", 0) > existing.get("confidence", 0):
                    by_name[name.lower()] = item
                elif existing:
                    existing.setdefault("sources", []).extend(item.get("sources", []))
            candidate_map[key] = list(by_name.values())

    def _apply_grounding(self, item: Dict[str, Any], docs: List[Dict[str, Any]]) -> None:
        name = item.get("name", "")
        if not name: return
        best = self._find_grounding(name, docs)
        if not best: return
        file_path, line_num, snippet = best
        item["confidence"] = min(round(float(item.get("confidence", 0.5)) + 0.15, 2), 0.99)
        item.setdefault("sources", []).append({
            "file": file_path, "line": line_num, "rule": "SRS_GROUNDING", "evidence": snippet[:140]
        })

    def _find_grounding(self, term: str, docs: List[Dict[str, Any]]) -> Optional[Tuple[str, int, str]]:
        needle = re.compile(rf"\b{re.escape(term)}\b", flags=re.IGNORECASE)
        for doc in docs:
            text = doc.get("content", "")
            if not text: continue
            for idx, line in enumerate(text.splitlines(), start=1):
                line_stripped = line.strip()
                if "do not use" in line_stripped.lower(): continue
                if needle.search(line_stripped):
                    return (doc.get("path", "srs_document"), idx, line_stripped)
        return None

    def _merge_candidates(self, existing: List[Dict[str, Any]], candidates: List[Dict[str, Any]], rule_name: str) -> None:
        existing_by_name = {item.get("name", "").lower(): item for item in existing}
        for cand in candidates:
            key = cand.get("name", "").lower()
            if not key: continue
            if key not in existing_by_name:
                existing.append(cand)
                existing_by_name[key] = cand
            else:
                current = existing_by_name[key]
                current["confidence"] = max(float(current.get("confidence", 0.5)), float(cand.get("confidence", 0.5)))
                current.setdefault("sources", []).extend(cand.get("sources", []))

    def _ensure_traceability(self, model_data: Dict[str, Any], srs_docs: Optional[List[Dict[str, Any]]]) -> None:
        if not srs_docs: srs_docs = []
        for ctx in model_data.get("bounded_contexts", []):
            ul = ctx.get("ubiquitous_language", {})
            for key in ("entities", "value_objects", "services", "aggregates"):
                for item in ul.get(key, []) or []:
                    if item.get("confidence") is None: item["confidence"] = 0.5
                    item["confidence"] = round(min(max(float(item["confidence"]), 0.0), 1.0), 2)
                    item.setdefault("sources", [])
                    if not item["sources"]:
                        grounded = self._find_grounding(item.get("name", ""), srs_docs)
                        if grounded:
                            file_path, line_num, snippet = grounded
                            item["sources"].append({
                                "file": file_path, "line": line_num, "rule": "SRS_ENTITY_MATCH", "evidence": snippet[:140]
                            })
                        else:
                            item["sources"].append({
                                "file": "generated", "line": 1, "rule": "LLM_SYNTHESIS", "evidence": "Generated by synthesis stage"
                            })