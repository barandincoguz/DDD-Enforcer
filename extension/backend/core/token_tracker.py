"""
Token Usage Tracker

Tracks and reports token usage across all LLM API calls for cost estimation.
Provides detailed statistics for UBMK presentation and research paper.

Supports multiple models with different pricing:
- gemini-2.5-flash (Domain Model Generation): Input $0.30/1M, Output $2.50/1M (includes thinking)
- gemini-2.5-flash-lite (Validation): Input $0.10/1M, Output $0.40/1M (includes thinking)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import json


# =============================================================================
# MODEL PRICING CONFIGURATION
# =============================================================================

# Gemini 2.5 Flash pricing (PAID tier - Jan 2026)
# Source: https://ai.google.dev/gemini-api/docs/pricing
FLASH_PRICING = {
    "input": 0.30 / 1_000_000,      # $0.30 per 1M tokens (text/image/video)
    "output": 2.50 / 1_000_000,     # $2.50 per 1M tokens (includes thinking tokens)
}

# Gemini 2.5 Flash-Lite pricing (PAID tier - Jan 2026)
FLASH_LITE_PRICING = {
    "input": 0.10 / 1_000_000,      # $0.10 per 1M tokens (text/image/video)
    "output": 0.40 / 1_000_000,     # $0.40 per 1M tokens (includes thinking tokens)
}

# Stage to model mapping
STAGE_MODEL_MAP = {
    # Domain Model Generation stages -> gemini-2.5-flash
    "Scout": "flash",
    "Architect": "flash", 
    "Specialist": "flash",
    "Synthesizer": "flash",
    # Validation stage -> gemini-2.5-flash-lite
    "Validator": "flash-lite",
}


@dataclass
class APICallRecord:
    """Single API call record with token usage."""
    
    timestamp: str
    stage: str  # e.g., "Scout", "Architect", "Specialist", "Validator"
    operation: str  # e.g., "extract_sentences", "identify_contexts", "validate_code"
    model: str  # "flash" or "flash-lite"
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float  # Cost in USD


@dataclass
class TokenUsageStats:
    """Aggregated token usage statistics."""
    
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_api_calls: int = 0
    
    # Per-model breakdown
    flash_prompt_tokens: int = 0
    flash_completion_tokens: int = 0
    flash_lite_prompt_tokens: int = 0
    flash_lite_completion_tokens: int = 0
    
    # Per-stage breakdown
    stage_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # Detailed call log
    call_history: List[APICallRecord] = field(default_factory=list)


class TokenTracker:
    """
    Global token usage tracker for cost estimation and reporting.
    
    Supports two models with different pricing:
    
    1. gemini-2.5-flash (Domain Model Generation):
       - Used by: Scout, Architect, Specialist, Synthesizer stages
       - Input: $0.30 per 1M tokens
       - Output: $2.50 per 1M tokens (includes thinking)
    
    2. gemini-2.5-flash-lite (Validation):
       - Used by: Validator stage
       - Input: $0.10 per 1M tokens
       - Output: $0.40 per 1M tokens (includes thinking)
    
    Source: https://ai.google.dev/gemini-api/docs/pricing
    
    Usage:
        tracker = TokenTracker.get_instance()
        tracker.track_api_call(response, stage="Scout", operation="extract")
        tracker.track_api_call(response, stage="Validator", operation="validate")
        report = tracker.get_report()
    """
    
    _instance: Optional['TokenTracker'] = None
    
    def __init__(self):
        self.stats = TokenUsageStats()
        self.session_start = datetime.now().isoformat()
    
    @classmethod
    def get_instance(cls) -> 'TokenTracker':
        """Singleton pattern to ensure single tracker across all modules."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset tracker (useful for testing)."""
        cls._instance = None
    
    def _get_model_for_stage(self, stage: str) -> str:
        """Determine which model is used for a given stage."""
        return STAGE_MODEL_MAP.get(stage, "flash-lite")
    
    def _calculate_call_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for a single API call based on model."""
        if model == "flash":
            pricing = FLASH_PRICING
            input_cost = prompt_tokens * pricing["input"]
            output_cost = completion_tokens * pricing["output"]  # Includes thinking tokens
        else:  # flash-lite
            pricing = FLASH_LITE_PRICING
            input_cost = prompt_tokens * pricing["input"]
            output_cost = completion_tokens * pricing["output"]
        
        return input_cost + output_cost
    
    def track_api_call(
        self, 
        response, 
        stage: str, 
        operation: str
    ) -> None:
        """
        Track token usage from a Gemini API response.
        
        Args:
            response: Gemini API response object with usage_metadata
            stage: Pipeline stage (Scout, Architect, Specialist, Synthesizer, Validator)
            operation: Specific operation name
        """
        # Extract usage metadata from Gemini response
        usage = response.usage_metadata
        
        # Safe extraction with None checks (some responses may have None values)
        prompt_tokens = getattr(usage, 'prompt_token_count', None) or 0
        completion_tokens = getattr(usage, 'candidates_token_count', None) or 0
        
        # Check for cached tokens (context caching feature)
        cached_tokens = getattr(usage, 'cached_content_token_count', None) or 0
        
        # Calculate billable tokens (excluding cached)
        billable_prompt_tokens = prompt_tokens - cached_tokens
        billable_total = billable_prompt_tokens + completion_tokens
        
        # Determine model based on stage
        model = self._get_model_for_stage(stage)
        
        # Calculate cost for this call
        call_cost = self._calculate_call_cost(model, billable_prompt_tokens, completion_tokens)
        
        # Log if there are cached tokens (not billed)
        if cached_tokens > 0:
            print(f"      💾 Cached: {cached_tokens:,} tokens (FREE) | Billable input: {billable_prompt_tokens:,}")
        
        # Create record (only billable tokens)
        record = APICallRecord(
            timestamp=datetime.now().isoformat(),
            stage=stage,
            operation=operation,
            model=model,
            prompt_tokens=billable_prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=billable_total,
            estimated_cost=round(call_cost, 8)
        )
        
        # Update aggregated stats
        self.stats.total_prompt_tokens += billable_prompt_tokens
        self.stats.total_completion_tokens += completion_tokens
        self.stats.total_tokens += billable_total
        self.stats.total_api_calls += 1
        
        # Update model-specific stats
        if model == "flash":
            self.stats.flash_prompt_tokens += billable_prompt_tokens
            self.stats.flash_completion_tokens += completion_tokens
        else:
            self.stats.flash_lite_prompt_tokens += billable_prompt_tokens
            self.stats.flash_lite_completion_tokens += completion_tokens
        
        # Update stage-specific stats
        if stage not in self.stats.stage_stats:
            self.stats.stage_stats[stage] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "call_count": 0,
                "model": model
            }
        
        self.stats.stage_stats[stage]["prompt_tokens"] += billable_prompt_tokens
        self.stats.stage_stats[stage]["completion_tokens"] += completion_tokens
        self.stats.stage_stats[stage]["total_tokens"] += billable_total
        self.stats.stage_stats[stage]["call_count"] += 1
        
        # Add to call history
        self.stats.call_history.append(record)
    
    def calculate_cost(self) -> Dict[str, float]:
        """
        Calculate estimated cost based on model-specific pricing.
        
        Returns:
            Dict with detailed cost breakdown by model and totals
        """
        # Flash model cost (Domain Model Generation)
        flash_input_cost = self.stats.flash_prompt_tokens * FLASH_PRICING["input"]
        flash_output_cost = self.stats.flash_completion_tokens * FLASH_PRICING["output"]
        flash_total = flash_input_cost + flash_output_cost
        
        # Flash-Lite model cost (Validation)
        lite_input_cost = self.stats.flash_lite_prompt_tokens * FLASH_LITE_PRICING["input"]
        lite_output_cost = self.stats.flash_lite_completion_tokens * FLASH_LITE_PRICING["output"]
        lite_total = lite_input_cost + lite_output_cost
        
        total_cost = flash_total + lite_total
        
        return {
            "flash_model": {
                "input_cost": round(flash_input_cost, 6),
                "output_cost": round(flash_output_cost, 6),
                "total_cost": round(flash_total, 6),
                "input_tokens": self.stats.flash_prompt_tokens,
                "output_tokens": self.stats.flash_completion_tokens
            },
            "flash_lite_model": {
                "input_cost": round(lite_input_cost, 6),
                "output_cost": round(lite_output_cost, 6),
                "total_cost": round(lite_total, 6),
                "input_tokens": self.stats.flash_lite_prompt_tokens,
                "output_tokens": self.stats.flash_lite_completion_tokens
            },
            "total_input_cost": round(flash_input_cost + lite_input_cost, 6),
            "total_output_cost": round(flash_output_cost + lite_output_cost, 6),
            "total_cost": round(total_cost, 6),
            "currency": "USD"
        }
    
    def get_report(self, detailed: bool = False) -> Dict:
        """
        Generate comprehensive token usage report.
        
        Args:
            detailed: Include per-call breakdown
            
        Returns:
            Dict with all statistics and cost estimation
        """
        cost = self.calculate_cost()
        
        report = {
            "session_start": self.session_start,
            "session_end": datetime.now().isoformat(),
            "summary": {
                "total_api_calls": self.stats.total_api_calls,
                "total_prompt_tokens": self.stats.total_prompt_tokens,
                "total_completion_tokens": self.stats.total_completion_tokens,
                "total_tokens": self.stats.total_tokens,
            },
            "model_usage": {
                "gemini-2.5-flash": {
                    "prompt_tokens": self.stats.flash_prompt_tokens,
                    "completion_tokens": self.stats.flash_completion_tokens,
                    "total_tokens": self.stats.flash_prompt_tokens + self.stats.flash_completion_tokens,
                    "stages": ["Scout", "Architect", "Specialist", "Synthesizer"]
                },
                "gemini-2.5-flash-lite": {
                    "prompt_tokens": self.stats.flash_lite_prompt_tokens,
                    "completion_tokens": self.stats.flash_lite_completion_tokens,
                    "total_tokens": self.stats.flash_lite_prompt_tokens + self.stats.flash_lite_completion_tokens,
                    "stages": ["Validator"]
                }
            },
            "cost_estimation": cost,
            "stage_breakdown": {}
        }
        
        # Add stage breakdown with model-specific pricing
        for stage, stats in self.stats.stage_stats.items():
            model = stats.get("model", self._get_model_for_stage(stage))
            
            if model == "flash":
                stage_input_cost = stats["prompt_tokens"] * FLASH_PRICING["input"]
                stage_output_cost = stats["completion_tokens"] * FLASH_PRICING["output"]
            else:
                stage_input_cost = stats["prompt_tokens"] * FLASH_LITE_PRICING["input"]
                stage_output_cost = stats["completion_tokens"] * FLASH_LITE_PRICING["output"]
            
            report["stage_breakdown"][stage] = {
                "model": f"gemini-2.5-{model}",
                "call_count": stats["call_count"],
                "prompt_tokens": stats["prompt_tokens"],
                "completion_tokens": stats["completion_tokens"],
                "total_tokens": stats["total_tokens"],
                "estimated_cost": round(stage_input_cost + stage_output_cost, 6)
            }
        
        # Add detailed call history if requested
        if detailed:
            report["call_history"] = [
                {
                    "timestamp": call.timestamp,
                    "stage": call.stage,
                    "operation": call.operation,
                    "model": f"gemini-2.5-{call.model}",
                    "prompt_tokens": call.prompt_tokens,
                    "completion_tokens": call.completion_tokens,
                    "total_tokens": call.total_tokens,
                    "estimated_cost": call.estimated_cost
                }
                for call in self.stats.call_history
            ]
        
        return report
    
    def print_summary(self):
        """Print formatted summary to console."""
        cost = self.calculate_cost()
        
        print("\n" + "="*70)
        print("📊 TOKEN USAGE & COST REPORT")
        print("="*70)
        print(f"  Total API Calls: {self.stats.total_api_calls}")
        print(f"  Total Tokens: {self.stats.total_tokens:,}")
        print(f"    ↳ Input:  {self.stats.total_prompt_tokens:,} tokens")
        print(f"    ↳ Output: {self.stats.total_completion_tokens:,} tokens")
        
        print("\n" + "-"*70)
        print("🤖 MODEL BREAKDOWN")
        print("-"*70)
        
        # Flash model (Domain Model)
        flash_total = self.stats.flash_prompt_tokens + self.stats.flash_completion_tokens
        if flash_total > 0:
            print(f"\n  gemini-2.5-flash (Domain Model Generation):")
            print(f"    Input:  {self.stats.flash_prompt_tokens:,} tokens @ $0.30/1M")
            print(f"    Output: {self.stats.flash_completion_tokens:,} tokens @ $2.50/1M (includes thinking)")
            print(f"    Cost:   ${cost['flash_model']['total_cost']:.6f}")
        
        # Flash-Lite model (Validation)
        lite_total = self.stats.flash_lite_prompt_tokens + self.stats.flash_lite_completion_tokens
        if lite_total > 0:
            print(f"\n  gemini-2.5-flash-lite (Validation):")
            print(f"    Input:  {self.stats.flash_lite_prompt_tokens:,} tokens @ $0.10/1M")
            print(f"    Output: {self.stats.flash_lite_completion_tokens:,} tokens @ $0.40/1M (includes thinking)")
            print(f"    Cost:   ${cost['flash_lite_model']['total_cost']:.6f}")
        
        print("\n" + "-"*70)
        print("💰 TOTAL COST ESTIMATION")
        print("-"*70)
        print(f"  Input Cost:  ${cost['total_input_cost']:.6f}")
        print(f"  Output Cost: ${cost['total_output_cost']:.6f}")
        print(f"  Total Cost:  ${cost['total_cost']:.6f} USD")
        
        if self.stats.stage_stats:
            print("\n" + "-"*70)
            print("📈 STAGE BREAKDOWN")
            print("-"*70)
            for stage, stats in self.stats.stage_stats.items():
                model = stats.get("model", self._get_model_for_stage(stage))
                
                if model == "flash":
                    stage_cost = (
                        stats["prompt_tokens"] * FLASH_PRICING["input"] +
                        stats["completion_tokens"] * FLASH_PRICING["output"]
                    )
                else:
                    stage_cost = (
                        stats["prompt_tokens"] * FLASH_LITE_PRICING["input"] +
                        stats["completion_tokens"] * FLASH_LITE_PRICING["output"]
                    )
                
                print(f"\n  {stage} (gemini-2.5-{model}):")
                print(f"    Calls: {stats['call_count']}")
                print(f"    Tokens: {stats['total_tokens']:,}")
                print(f"    Cost: ${stage_cost:.6f}")
        
        print("="*70 + "\n")
    
    def export_to_json(self, filepath: str, detailed: bool = True):
        """Export report to JSON file for analysis."""
        report = self.get_report(detailed=detailed)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📄 Token usage report exported to: {filepath}")
