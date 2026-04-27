import textwrap

from core.code_parser import build_advanced_validation_payload
from core.llm_client import LLMClient
from core.parser import CodeParser
from main import _needs_llm_advanced_checks


COMPLEX_CODE = textwrap.dedent(
    """
    from dataclasses import dataclass
    from ..sales.domain import OrderPlaced as SalesOrderPlaced
    import billing.service as billing_service

    @dataclass(frozen=True)
    class Order(models.BaseAggregate):
        kind: str = "sales"

        def __init__(self, order_id: str, *, status: str = "draft") -> None:
            self.order_id: str = order_id
            self.status = status
            self.events = []

        async def publish(self, bus, *, event_name: str = "OrderPlaced") -> None:
            self.total += 1
            self.events.append("OrderPlaced")
            bus.publish("OrderPlaced", source="order")

            def nested_helper(amount: int = 1) -> int:
                self.status = "ready"
                return amount

            nested_helper()

    async def sync_projection(order_id: str, *, notify: bool = True) -> None:
        payload: dict[str, str] = {}
        emitter.emit("ProjectionUpdated", source="projection")
        return None
    """
)


def test_parser_preserves_public_shape_and_adds_richer_metadata():
    parser = CodeParser()

    result = parser.parse_code(COMPLEX_CODE, "order.py")

    assert set(result) == {
        "filename",
        "classes",
        "imports",
        "functions",
        "assignments",
        "function_calls",
    }
    assert result["filename"] == "order.py"

    order_class = result["classes"][0]
    assert order_class["name"] == "Order"
    assert order_class["bases"] == ["models.BaseAggregate"]
    assert order_class["base_names"] == ["BaseAggregate"]
    assert order_class["decorators"] == ["dataclass(frozen=True)"]
    assert order_class["methods"] == ["__init__", "publish"]
    assert order_class["class_attributes"] == ["kind"]
    assert set(order_class["instance_attributes"]) >= {
        "order_id",
        "status",
        "events",
        "total",
    }
    assert order_class["is_dataclass"] is True
    assert order_class["is_frozen"] is True
    assert set(order_class["mutation_methods"]) >= {"__init__", "publish"}

    imports = result["imports"]
    assert imports[0]["module"] == "dataclasses"
    assert imports[0]["type"] == "from"
    assert imports[0]["names"] == ["dataclass"]
    assert imports[1]["module"] == "sales.domain"
    assert imports[1]["level"] == 2
    assert imports[1]["members"] == [{"name": "OrderPlaced", "asname": "SalesOrderPlaced"}]
    assert imports[2]["module"] == "billing.service"
    assert imports[2]["asname"] == "billing_service"
    assert imports[2]["imported_modules"] == ["billing.service"]


def test_parser_tracks_async_nested_functions_and_assignment_variants():
    parser = CodeParser()

    result = parser.parse_code(COMPLEX_CODE, "order.py")
    functions = {record["name"]: record for record in result["functions"]}
    assignments = {record["target"]: record for record in result["assignments"] if record["target"]}
    calls = {record["function"]: record for record in result["function_calls"] if record["function"]}

    publish = functions["publish"]
    status_param = next(
        parameter
        for parameter in functions["__init__"]["parameters"]
        if parameter["name"] == "status"
    )
    nested_helper = functions["nested_helper"]
    sync_projection = functions["sync_projection"]

    assert publish["is_async"] is True
    assert publish["return_annotation"] == "None"
    assert status_param["kind"] == "keyword_only"
    assert status_param["has_default"] is True
    assert status_param["default"] == "'draft'"
    assert nested_helper["in_function"] == "publish"
    assert nested_helper["nesting_level"] == 1
    assert sync_projection["is_async"] is True

    assert assignments["self.order_id"]["annotation"] == "str"
    assert assignments["self.order_id"]["value_type"] == "variable"
    assert assignments["self.order_id"]["value_shape"] == "name:order_id"
    assert assignments["self.total"]["operation"] == "+="
    assert assignments["payload"]["annotation"] == "dict[str, str]"
    assert assignments["payload"]["value_type"] == "dict"

    assert calls["bus.publish"]["receiver"] == "bus"
    assert calls["bus.publish"]["kwargs"] == {"source": "order"}
    assert calls["emitter.emit"]["args"] == ["ProjectionUpdated"]
    assert calls["nested_helper"]["in_function"] == "publish"


def test_needs_llm_advanced_checks_prefers_filtered_rich_signals():
    parser = CodeParser()
    low_signal = parser.parse_code("import datetime\n\nclass Clock:\n    pass\n", "clock.py")

    assert _needs_llm_advanced_checks(low_signal) is False


def test_needs_llm_advanced_checks_keeps_legacy_truthy_fallback():
    legacy_ast = {
        "filename": "legacy.py",
        "classes": [],
        "functions": [],
        "imports": [{"module": "sales", "type": "import", "line": 1}],
        "assignments": [],
        "function_calls": [],
    }

    assert _needs_llm_advanced_checks(legacy_ast) is True


def test_advanced_payload_and_prompt_are_compact_and_relevant():
    parser = CodeParser()
    ast_data = parser.parse_code(COMPLEX_CODE, "order.py")
    payload = build_advanced_validation_payload(ast_data)
    llm = object.__new__(LLMClient)

    prompt = llm._build_advanced_prompt(
        ast_data,
        {
            "bounded_contexts": [
                {
                    "context_name": "Sales",
                    "allowed_dependencies": ["Billing"],
                    "ubiquitous_language": {
                        "value_objects": [{"name": "Money"}],
                        "domain_events": ["OrderPlaced", "ProjectionUpdated"],
                    },
                }
            ]
        },
    )

    assert payload["imports"]
    assert payload["imports"][0]["module"] == "sales.domain"
    assert "dataclasses" not in prompt
    assert "billing.service" in prompt
    assert "bus.publish" in prompt
    assert "emitter.emit" in prompt
    assert "nested_helper" not in prompt
