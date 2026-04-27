from textwrap import dedent

from core.AST.ast_model_signals import ASTModelSignalExtractor
from core.AST.ast_signal_grounding import rank_groundings
from core.schemas import BoundedContext, DomainModel, Entity, ProjectMetadata, UbiquitousLanguage


def _write_module(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content), encoding="utf-8")
    return path


def _build_context(name, entities=None):
    return BoundedContext(
        context_name=name,
        description=f"{name} responsibilities",
        allowed_dependencies=[],
        ubiquitous_language=UbiquitousLanguage(
            entities=entities or [],
            value_objects=[],
            services=[],
            aggregates=[],
            domain_events=[],
        ),
    )


def _build_model(*contexts):
    return DomainModel(
        project_name="Test",
        project_metadata=ProjectMetadata(
            version="1.0.0",
            generated_at="2026-03-13",
            description="test",
        ),
        bounded_contexts=list(contexts),
        global_rules=None,
    )


def test_detects_domain_event_from_publisher_call(tmp_path):
    workspace = tmp_path / "workspace"
    _write_module(
        workspace / "orders.py",
        """
        class OrderService:
            def __init__(self, event_bus):
                self.event_bus = event_bus

            def complete(self):
                self.event_bus.publish("OrderPlaced")
        """,
    )

    model = _build_model()
    enriched = ASTModelSignalExtractor().enrich_domain_model(model, str(workspace))

    context = enriched.bounded_contexts[0].ubiquitous_language
    assert "OrderPlaced" in (context.domain_events or [])


def test_filters_infrastructure_false_positives(tmp_path):
    sample = _write_module(
        tmp_path / "infra.py",
        """
        from dataclasses import dataclass

        @dataclass
        class UserDTO:
            username: str
            password: str

        class CustomerRepository:
            pass

        class ApiController:
            pass

        class StringHelper:
            pass

        class AppConfig:
            pass

        class DomainException(Exception):
            pass
        """,
    )

    candidates = ASTModelSignalExtractor().extract_candidates([str(sample)])
    all_names = {
        item["name"]
        for bucket in candidates.values()
        for item in bucket
    }

    assert "UserDTO" not in all_names
    assert "CustomerRepository" not in all_names
    assert "ApiController" not in all_names
    assert "StringHelper" not in all_names
    assert "AppConfig" not in all_names
    assert "DomainException" not in all_names


def test_context_assignment_prefers_grounded_context(tmp_path):
    workspace = tmp_path / "workspace"
    _write_module(
        workspace / "billing" / "invoice.py",
        """
        class Invoice:
            def __init__(self, invoice_id, amount):
                self.invoice_id = invoice_id
                self.amount = amount
        """,
    )

    model = _build_model(_build_context("BillingContext"), _build_context("SalesContext"))
    enriched = ASTModelSignalExtractor().enrich_domain_model(
        model,
        str(workspace),
        srs_docs=[{"path": "srs.txt", "content": "Billing context: Invoice must include amount."}],
    )

    billing_entities = [entity.name for entity in enriched.bounded_contexts[0].ubiquitous_language.entities]
    sales_entities = [entity.name for entity in enriched.bounded_contexts[1].ubiquitous_language.entities]

    assert "Invoice" in billing_entities
    assert "Invoice" not in sales_entities


def test_ambiguous_context_candidate_is_not_forced(tmp_path):
    workspace = tmp_path / "workspace"
    _write_module(
        workspace / "billing_shipping" / "invoice_processor.py",
        """
        class InvoiceProcessor:
            def __init__(self, invoice_repository):
                self.invoice_repository = invoice_repository

            def process(self):
                return True
        """,
    )

    shared_entity = Entity(name="Invoice", description="invoice")
    model = _build_model(
        _build_context("BillingContext", entities=[shared_entity]),
        _build_context("ShippingContext", entities=[shared_entity]),
    )

    enriched = ASTModelSignalExtractor().enrich_domain_model(
        model,
        str(workspace),
        srs_docs=[{"path": "srs.txt", "content": "Billing and Shipping both process invoices."}],
    )

    billing_services = enriched.bounded_contexts[0].ubiquitous_language.services or []
    shipping_services = enriched.bounded_contexts[1].ubiquitous_language.services or []

    assert all(service.name != "InvoiceProcessor" for service in billing_services)
    assert all(service.name != "InvoiceProcessor" for service in shipping_services)


def test_low_confidence_ast_only_candidate_is_not_merged(tmp_path):
    workspace = tmp_path / "workspace"
    _write_module(
        workspace / "billing" / "invoice.py",
        """
        class Invoice:
            def __init__(self, invoice_id):
                self.invoice_id = invoice_id
        """,
    )

    model = _build_model(_build_context("BillingContext"))
    enriched = ASTModelSignalExtractor().enrich_domain_model(model, str(workspace), srs_docs=[])

    billing_entities = enriched.bounded_contexts[0].ubiquitous_language.entities
    assert all(entity.name != "Invoice" for entity in billing_entities)


def test_rank_groundings_handles_exact_plural_and_negative_guidance():
    docs = [
        {
            "path": "srs.txt",
            "content": "\n".join(
                [
                    "Invoices must include amount.",
                    "Invoice must include currency.",
                    "Do not use InvoiceRecord in the domain model.",
                ]
            ),
        }
    ]

    matches = rank_groundings("Invoice", docs, limit=3)
    snippets = [match.snippet for match in matches]

    assert any("Invoices must include amount." == snippet for snippet in snippets)
    assert any("Invoice must include currency." == snippet for snippet in snippets)
    assert all("Do not use InvoiceRecord" not in snippet for snippet in snippets)


def test_extract_candidates_deduplicates_across_files_and_combines_sources(tmp_path):
    first = _write_module(
        tmp_path / "order_a.py",
        """
        class Order:
            def __init__(self, order_id, total):
                self.order_id = order_id
                self.total = total
        """,
    )
    second = _write_module(
        tmp_path / "order_b.py",
        """
        class Order:
            def __init__(self, order_id, total):
                self.order_id = order_id
                self.total = total

            def cancel(self):
                return True
        """,
    )

    candidates = ASTModelSignalExtractor().extract_candidates([str(first), str(second)], grounding_docs=[])
    order_entities = [candidate for candidate in candidates["entities"] if candidate["name"] == "Order"]

    assert len(order_entities) == 1
    assert len(order_entities[0]["sources"]) >= 2
