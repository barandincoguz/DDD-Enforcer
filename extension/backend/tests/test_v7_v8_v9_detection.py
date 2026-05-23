"""WP-CORE-33 — V7 ACL + V8 Specification + V9 Service-kind tests (RED phase).

Closes three DDD-pattern detection blind spots highlighted in the
iteration-18 pipeline-hardening agent findings:

* **V7 Anti-Corruption Layer** — classes that translate between an
  internal bounded context and an external system (Translator, Adapter,
  Mapper, Gateway, ACL suffixes plus translate_/convert_/to_domain
  methods).  Previously fell through every scorer and was never emitted
  as a first-class candidate.

* **V8 Specification pattern** — predicate-as-object classes
  (`is_satisfied_by`) used to encapsulate domain rules.  Same problem:
  silently dropped by the legacy scorer chain.

* **V9 Service-kind discriminator** — `_score_service` already detected
  *services* but never told downstream consumers whether a candidate is
  a pure **domain service** (no infra deps), an **application service**
  (orchestrates use cases with repositories injected), or an
  **infrastructure service** (wraps external clients).  Adds an
  optional `kind` field on `Service` + a deterministic classifier.

All new candidate types are emitted by `ASTModelSignalExtractor` with
their own dedicated bucket in the public payload, exactly mirroring the
WP-CORE-22 Repository/Factory pattern.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.AST.ast_model_signals import ASTModelSignalExtractor
from core.AST.ast_signal_enrichment import CANDIDATE_TYPES
from core.schemas import (
    AntiCorruptionLayer,
    Service,
    Specification,
    UbiquitousLanguage,
)


class _FixtureWorkspace:
    """Mixin that gives each test a temp dir + a `_write` helper."""

    def setUp(self) -> None:  # type: ignore[override]
        self.test_dir = tempfile.mkdtemp()
        self.extractor = ASTModelSignalExtractor()

    def tearDown(self) -> None:  # type: ignore[override]
        shutil.rmtree(self.test_dir)

    def _write(self, filename: str, content: str) -> str:
        path = os.path.join(self.test_dir, filename)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return path


# =============================================================================
# V7 — Anti-Corruption Layer detection
# =============================================================================


class TestAntiCorruptionLayerDetection(_FixtureWorkspace, unittest.TestCase):
    def test_t_acl_1_translator_suffix_with_translate_methods(self) -> None:
        """T-ACL-1: `*Translator` + translate_/to_domain/from_external methods → ACL."""
        code = """
class StripePaymentTranslator:
    def __init__(self, stripe_client):
        self.stripe_client = stripe_client

    def translate_to_domain(self, stripe_charge):
        return Payment(amount=stripe_charge["amount"])

    def from_external(self, payload):
        return self.translate_to_domain(payload)
"""
        path = self._write("acl_translator.py", code)
        results = self.extractor.extract_candidates([path])
        names = [item["name"] for item in results.get("anti_corruption_layers", [])]
        self.assertIn("StripePaymentTranslator", names)

    def test_t_acl_2_adapter_suffix_with_convert_methods(self) -> None:
        """T-ACL-2: `*Adapter` with `convert_*` methods → ACL."""
        code = """
class LegacyOrderAdapter:
    def __init__(self, legacy_api):
        self.legacy_api = legacy_api

    def convert_legacy_order(self, raw):
        return Order(id=raw["order_id"])

    def adapt(self, raw):
        return self.convert_legacy_order(raw)
"""
        path = self._write("acl_adapter.py", code)
        results = self.extractor.extract_candidates([path])
        names = [item["name"] for item in results.get("anti_corruption_layers", [])]
        self.assertIn("LegacyOrderAdapter", names)

    def test_t_acl_3_gateway_suffix_with_to_domain_methods(self) -> None:
        """T-ACL-3: `*Gateway` with explicit `to_domain` translation → ACL."""
        code = """
class ShopifyGateway:
    def __init__(self, sdk):
        self.sdk = sdk

    def to_domain(self, shopify_product):
        return Product(sku=shopify_product["sku"])
"""
        path = self._write("acl_gateway.py", code)
        results = self.extractor.extract_candidates([path])
        names = [item["name"] for item in results.get("anti_corruption_layers", [])]
        self.assertIn("ShopifyGateway", names)

    def test_t_acl_4_plain_entity_not_detected_as_acl(self) -> None:
        """T-ACL-4: ordinary domain entity is NOT flagged as ACL."""
        code = """
class Customer:
    def __init__(self, customer_id, name):
        self.customer_id = customer_id
        self.name = name
"""
        path = self._write("entity.py", code)
        results = self.extractor.extract_candidates([path])
        names = [item["name"] for item in results.get("anti_corruption_layers", [])]
        self.assertNotIn("Customer", names)

    def test_t_acl_5_application_service_not_detected_as_acl(self) -> None:
        """T-ACL-5: a Service without translate-shaped methods is NOT an ACL.

        `*Service` deliberately lives outside ACL_SUFFIXES.  Even with an
        external client injected, no translate/convert/to_domain method
        means it stays a Service, not an ACL.
        """
        code = """
class PaymentService:
    def __init__(self, repo):
        self.repo = repo

    def process(self, order_id):
        order = self.repo.find_by_id(order_id)
        return order
"""
        path = self._write("svc_no_acl.py", code)
        results = self.extractor.extract_candidates([path])
        names = [item["name"] for item in results.get("anti_corruption_layers", [])]
        self.assertNotIn("PaymentService", names)

    def test_t_acl_6_payload_carries_description(self) -> None:
        """T-ACL-6: payload exposes name + description + confidence."""
        code = """
class StripeChargeMapper:
    def __init__(self, client):
        self.client = client

    def translate(self, raw):
        return Charge()

    def convert_charge(self, raw):
        return Charge()
"""
        path = self._write("acl_mapper.py", code)
        results = self.extractor.extract_candidates([path])
        acls = results.get("anti_corruption_layers", [])
        target = next((a for a in acls if a["name"] == "StripeChargeMapper"), None)
        self.assertIsNotNone(target)
        self.assertGreaterEqual(target["confidence"], 0.6)
        self.assertIn("sources", target)


# =============================================================================
# V8 — Specification pattern detection
# =============================================================================


class TestSpecificationDetection(_FixtureWorkspace, unittest.TestCase):
    def test_t_spec_1_specification_suffix_with_is_satisfied_by(self) -> None:
        """T-SPEC-1: `*Specification` + `is_satisfied_by(...)` → specification."""
        code = """
class PremiumCustomerSpecification:
    def is_satisfied_by(self, customer):
        return customer.tier == "premium"
"""
        path = self._write("spec_basic.py", code)
        results = self.extractor.extract_candidates([path])
        names = [item["name"] for item in results.get("specifications", [])]
        self.assertIn("PremiumCustomerSpecification", names)

    def test_t_spec_2_spec_short_suffix_accepted(self) -> None:
        """T-SPEC-2: short `*Spec` suffix also accepted."""
        code = """
class OverdueInvoiceSpec:
    def is_satisfied_by(self, invoice):
        return invoice.is_past_due()
"""
        path = self._write("spec_short.py", code)
        results = self.extractor.extract_candidates([path])
        names = [item["name"] for item in results.get("specifications", [])]
        self.assertIn("OverdueInvoiceSpec", names)

    def test_t_spec_3_combinator_methods_score(self) -> None:
        """T-SPEC-3: a specification with `and_`/`or_`/`not_` combinators is detected
        even at modest evidence weight."""
        code = """
class ActiveOrderSpec:
    def is_satisfied_by(self, order):
        return order.is_active()

    def and_(self, other):
        return CompositeSpec(self, other, "and")

    def or_(self, other):
        return CompositeSpec(self, other, "or")
"""
        path = self._write("spec_combinator.py", code)
        results = self.extractor.extract_candidates([path])
        names = [item["name"] for item in results.get("specifications", [])]
        self.assertIn("ActiveOrderSpec", names)

    def test_t_spec_4_random_class_not_detected_as_spec(self) -> None:
        """T-SPEC-4: a plain dataclass without `is_satisfied_by` is NOT a spec."""
        code = """
class Amount:
    def __init__(self, value, currency):
        self.value = value
        self.currency = currency
"""
        path = self._write("not_spec.py", code)
        results = self.extractor.extract_candidates([path])
        names = [item["name"] for item in results.get("specifications", [])]
        self.assertNotIn("Amount", names)

    def test_t_spec_5_spec_with_heavy_di_demoted(self) -> None:
        """T-SPEC-5: a class with `Spec` suffix but heavy DI (>=2 repo deps) and
        NO `is_satisfied_by` is NOT detected — would shift to ApplicationService."""
        code = """
class OrderSpec:
    def __init__(self, order_repo, customer_repo, audit_log):
        self.order_repo = order_repo
        self.customer_repo = customer_repo
        self.audit_log = audit_log

    def process(self):
        return self.order_repo.find_all()
"""
        path = self._write("fake_spec.py", code)
        results = self.extractor.extract_candidates([path])
        names = [item["name"] for item in results.get("specifications", [])]
        self.assertNotIn("OrderSpec", names)


# =============================================================================
# V9 — Service-kind discriminator
# =============================================================================


class TestServiceKindDiscriminator(_FixtureWorkspace, unittest.TestCase):
    def _find_service(self, results, name):
        return next(
            (s for s in results.get("services", []) if s["name"] == name),
            None,
        )

    def test_t_svc_kind_1_domain_service_no_deps(self) -> None:
        """T-SVC-KIND-1: Service with zero injected deps and pure-logic methods → domain."""
        code = """
class PricingService:
    def calculate(self, base, modifier):
        return base * modifier

    def discount(self, base, percent):
        return base * (1 - percent / 100)
"""
        path = self._write("pricing.py", code)
        results = self.extractor.extract_candidates([path])
        target = self._find_service(results, "PricingService")
        self.assertIsNotNone(target)
        self.assertEqual(target["kind"], "domain")

    def test_t_svc_kind_2_application_service_with_repo(self) -> None:
        """T-SVC-KIND-2: Service injecting a repository → application."""
        code = """
class PlaceOrderService:
    def __init__(self, order_repo):
        self.order_repo = order_repo

    def execute(self, payload):
        order = self.order_repo.save(payload)
        return order
"""
        path = self._write("place_order.py", code)
        results = self.extractor.extract_candidates([path])
        target = self._find_service(results, "PlaceOrderService")
        self.assertIsNotNone(target)
        self.assertEqual(target["kind"], "application")

    def test_t_svc_kind_3_infrastructure_service_with_only_clients(self) -> None:
        """T-SVC-KIND-3: Service whose deps are only external clients/gateways → infrastructure."""
        code = """
class EmailService:
    def __init__(self, smtp_client, sendgrid_gateway):
        self.smtp_client = smtp_client
        self.sendgrid_gateway = sendgrid_gateway

    def send(self, message):
        self.smtp_client.deliver(message)
"""
        path = self._write("email.py", code)
        results = self.extractor.extract_candidates([path])
        target = self._find_service(results, "EmailService")
        self.assertIsNotNone(target)
        self.assertEqual(target["kind"], "infrastructure")

    def test_t_svc_kind_4_mixed_repo_and_gateway_resolves_to_application(self) -> None:
        """T-SVC-KIND-4: repository + gateway in same constructor → application (repo wins)."""
        code = """
class CheckoutService:
    def __init__(self, order_repo, payment_gateway):
        self.order_repo = order_repo
        self.payment_gateway = payment_gateway

    def execute(self, order_id):
        order = self.order_repo.find_by_id(order_id)
        return self.payment_gateway.charge(order)
"""
        path = self._write("checkout.py", code)
        results = self.extractor.extract_candidates([path])
        target = self._find_service(results, "CheckoutService")
        self.assertIsNotNone(target)
        self.assertEqual(target["kind"], "application")

    def test_t_svc_kind_5_kind_is_persistable_across_to_public_dict(self) -> None:
        """T-SVC-KIND-5: the `kind` field is part of the public payload structure."""
        code = """
class TaxService:
    def compute(self, amount):
        return amount * 0.18
"""
        path = self._write("tax.py", code)
        results = self.extractor.extract_candidates([path])
        target = self._find_service(results, "TaxService")
        self.assertIsNotNone(target)
        self.assertIn("kind", target)
        self.assertIn(target["kind"], {"domain", "application", "infrastructure"})


# =============================================================================
# Schema + enrichment wiring
# =============================================================================


class TestSchemaIntegration(unittest.TestCase):
    def test_t_schema_1_anti_corruption_layer_pydantic_instantiation(self) -> None:
        """T-SCHEMA-1: `AntiCorruptionLayer` accepts minimal valid payload."""
        acl = AntiCorruptionLayer(
            name="StripeAdapter",
            description="Maps Stripe charges to domain Payment",
            confidence=0.7,
        )
        self.assertEqual(acl.name, "StripeAdapter")
        self.assertEqual(acl.confidence, 0.7)
        self.assertEqual(acl.sources, [])

    def test_t_schema_2_specification_pydantic_instantiation(self) -> None:
        """T-SCHEMA-2: `Specification` accepts minimal valid payload."""
        spec = Specification(
            name="PremiumCustomerSpecification",
            description="Premium tier predicate",
            confidence=0.65,
        )
        self.assertEqual(spec.name, "PremiumCustomerSpecification")
        self.assertEqual(spec.confidence, 0.65)

    def test_t_schema_3_service_kind_optional_default_none(self) -> None:
        """T-SCHEMA-3: existing Service payloads without `kind` still validate."""
        svc = Service(
            name="LegacyService",
            description="Legacy payload without explicit kind.",
        )
        self.assertIsNone(svc.kind)

    def test_t_schema_4_service_kind_literal_accepted(self) -> None:
        """T-SCHEMA-4: Service.kind accepts the three documented literals."""
        for kind in ("domain", "application", "infrastructure"):
            svc = Service(name=f"S_{kind}", description="x", kind=kind)
            self.assertEqual(svc.kind, kind)

    def test_t_schema_5_ubiquitous_language_acl_and_spec_fields(self) -> None:
        """T-SCHEMA-5: `UbiquitousLanguage` exposes optional ACL + Specification fields."""
        ul = UbiquitousLanguage(
            entities=[],
            value_objects=None,
            services=None,
            aggregates=None,
            domain_events=None,
            anti_corruption_layers=[
                AntiCorruptionLayer(name="StripeAdapter"),
            ],
            specifications=[
                Specification(name="PremiumCustomerSpecification"),
            ],
        )
        assert ul.anti_corruption_layers is not None
        assert ul.specifications is not None
        self.assertEqual(len(ul.anti_corruption_layers), 1)
        self.assertEqual(len(ul.specifications), 1)

    def test_t_schema_6_candidate_types_extended(self) -> None:
        """T-SCHEMA-6: enrichment CANDIDATE_TYPES tuple includes both new buckets."""
        self.assertIn("anti_corruption_layers", CANDIDATE_TYPES)
        self.assertIn("specifications", CANDIDATE_TYPES)


if __name__ == "__main__":
    unittest.main()
