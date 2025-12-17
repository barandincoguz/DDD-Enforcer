"""
DDD-Enforcer Configuration

Central configuration file for all application settings.
Modify these values to customize the behavior of the system.
"""

from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).parent.absolute()
INPUTS_DIR = BASE_DIR / "inputs"
DOMAIN_DIR = BASE_DIR / "domain"
DOMAIN_MODEL_PATH = DOMAIN_DIR / "model.json"


# =============================================================================
# RAG PIPELINE CONFIGURATION
# =============================================================================

class RAGConfig:
    """Configuration for the RAG (Retrieval Augmented Generation) pipeline."""

    # Storage
    PERSIST_DIRECTORY: str = str(BASE_DIR / "data" / "chroma_db")
    COLLECTION_NAME: str = "srs_documents"
    DISTANCE_METRIC: str = "cosine"

    # Chunking
    CHUNK_SIZE: int = 250
    CHUNK_OVERLAP: int = 30
    CHUNKS_PER_PAGE: int = 4

    # Retrieval
    TOP_K: int = 3
    MIN_RELEVANCE_SCORE: float = 0.3

    # Summary
    MAX_SUMMARY_LENGTH: int = 100

    # Bounded context keywords for metadata extraction
    BOUNDED_CONTEXT_KEYWORDS: list = [
        "Customer Management", "Order Processing",
        "Inventory Management", "Payment Processing",
        "Product Catalog", "Shopping Cart", "Shipping",
        "User Management", "Account Management"
    ]

    # Entity names to extract from text
    ENTITY_NAMES: list = [
        "Customer", "Order", "Product", "Payment", "Address",
        "OrderItem", "StockLevel", "Category", "Wishlist", "Money",
        "Invoice", "Shipment", "Cart", "Discount", "Coupon",
        "User", "Account", "Transaction", "Inventory", "Supplier"
    ]

    # Synonym terms (terms that should be avoided)
    SYNONYM_TERMS: list = [
        "Client", "User", "Buyer", "Shopper", "Consumer",
        "Item", "Good", "Merchandise", "Article",
        "Purchase", "Transaction", "Sale", "Charge",
        "Basket", "Bag"
    ]

    # Banned generic terms
    BANNED_TERMS: list = [
        "Manager", "Helper", "Util", "Utility", "Data", "Info",
        "Handler", "Processor", "Controller", "Bean", "DTO"
    ]

    # Chunk classification keywords
    GLOSSARY_KEYWORDS: list = ["glossary", "terminology", "definitions"]
    DEPENDENCY_KEYWORDS: list = ["dependencies", "relationship", "integration"]
    DOMAIN_RULE_KEYWORDS: list = ["important:", "should not be used", "must not", "must be"]
    GLOBAL_RULE_KEYWORDS: list = ["banned", "forbidden", "prohibited"]


# =============================================================================
# LLM CONFIGURATION
# =============================================================================

class AnalyzerConfig:
    """Configuration for the Code Analyzer LLM client (violation detection)."""

    MODEL_NAME: str = "gemini-2.5-flash-lite"
    RESPONSE_MIME_TYPE: str = "application/json"
    TEMPERATURE: float = 0.05
    MAX_OUTPUT_TOKENS: int = 8000


class ArchitectConfig:
    """Configuration for the Domain Architect LLM client (domain model generation)."""

    MODEL_NAME: str = "gemini-2.5-flash"
    RESPONSE_MIME_TYPE: str = "application/json"
    TEMPERATURE: float = 0.10 # Slight randomness for better naming choices, still highly consistent
    MAX_OUTPUT_TOKENS: int =10000 

# =============================================================================
# SERVER CONFIGURATION
# =============================================================================

class ServerConfig:
    """Configuration for the FastAPI server."""

    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = False


# =============================================================================
# DOCUMENT PARSER CONFIGURATION
# =============================================================================

class ParserConfig:
    """Configuration for document parsing."""

    SUPPORTED_EXTENSIONS: list = [".pdf", ".docx", ".txt"]
    MAX_DOCUMENT_SIZE: int = 1_000_000
    SECTION_HEADER_PATTERN: str = r'^(\d+\.?\d*\.?)\s+(.+?)$'
    MAX_SECTION_HEADER_LENGTH: int = 100


# =============================================================================
# VIOLATION DETECTION CONFIGURATION
# =============================================================================

class ViolationConfig:
    """Configuration for DDD violation detection."""

    BANNED_GLOBAL_TERMS: list = [
        "Manager", "Helper", "Util", "Utility", "Data", "Info",
        "Handler", "Processor", "Controller", "Bean", "DTO"
    ]

    ALLOWED_PATTERNS: list = [
        "Repository", "Service", "Factory", "Gateway",
        "Aggregate", "Specification", "Event", "ValueObject", "Entity"
    ]

    COMMON_SYNONYMS: dict = {
        "Customer": ["Client", "User", "Buyer", "Shopper", "Consumer"],
        "Order": ["Purchase", "Transaction", "Sale", "Acquisition"],
        "Product": ["Item", "Good", "Merchandise", "Article"],
        "Payment": ["Transaction", "Charge", "Bill", "Invoice"],
    }


# =============================================================================
# EXTENSION CONFIGURATION
# =============================================================================

class ExtensionConfig:
    """Reference configuration for VS Code extension settings."""

    BACKEND_URL: str = f"http://{ServerConfig.HOST}:{ServerConfig.PORT}"
    VALIDATE_ENDPOINT: str = f"{BACKEND_URL}/validate"
    RAG_STATS_ENDPOINT: str = f"{BACKEND_URL}/rag/stats"
    RAG_SEARCH_ENDPOINT: str = f"{BACKEND_URL}/rag/search"
