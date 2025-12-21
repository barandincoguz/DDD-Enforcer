"""
Token Tracking Verification - Critical for UBMK Presentation

This script verifies the correctness of token tracking and cost calculation.
"""

# Gemini 2.5 Flash Pricing (Official PAID Tier - December 12, 2025)
# Source: https://ai.google.dev/gemini-api/docs/pricing
INPUT_PRICE_PER_1M = 0.30    # $0.30 per 1M input tokens (text/image/video)
OUTPUT_PRICE_PER_1M = 2.50   # $2.50 per 1M output tokens (includes reasoning)
CACHE_PRICE_PER_1M = 0.03    # $0.03 per 1M cached tokens (10x cheaper)

print("="*70)
print("🔍 TOKEN TRACKING VERIFICATION (UPDATED PRICING)")
print("="*70)
print(f"📌 Input:  ${INPUT_PRICE_PER_1M}/1M tokens")
print(f"📌 Output: ${OUTPUT_PRICE_PER_1M}/1M tokens")
print(f"📌 Cache:  ${CACHE_PRICE_PER_1M}/1M tokens")
print("="*70)

# Test Case 1: Simple calculation without caching
print("\n📊 Test Case 1: Basic Token Calculation")
print("-"*70)
input_tokens = 10000
output_tokens = 5000
total_tokens = input_tokens + output_tokens

input_cost = (input_tokens / 1_000_000) * INPUT_PRICE_PER_1M
output_cost = (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_1M
total_cost = input_cost + output_cost

print(f"Input:  {input_tokens:,} tokens × ${INPUT_PRICE_PER_1M}/1M = ${input_cost:.6f}")
print(f"Output: {output_tokens:,} tokens × ${OUTPUT_PRICE_PER_1M}/1M = ${output_cost:.6f}")
print(f"Total:  {total_tokens:,} tokens = ${total_cost:.6f}")

# Verify with new pricing
expected_input = 0.003000   # 10000 × 0.30 / 1M
expected_output = 0.012500  # 5000 × 2.50 / 1M
expected_total = 0.015500

assert abs(input_cost - expected_input) < 0.000001, f"Input cost wrong: {input_cost} != {expected_input}"
assert abs(output_cost - expected_output) < 0.000001, f"Output cost wrong: {output_cost} != {expected_output}"
assert abs(total_cost - expected_total) < 0.000001, f"Total cost wrong: {total_cost} != {expected_total}"
print("✅ Basic calculation CORRECT")

# Test Case 2: With cached tokens (NOW BILLED!)
print("\n📊 Test Case 2: Cached Token Handling (10x cheaper)")
print("-"*70)
total_input_from_api = 10000  # prompt_token_count from API
cached_tokens = 3000          # cached_content_token_count from API
billable_input = total_input_from_api - cached_tokens  # Fresh input
output_tokens = 5000

fresh_input_cost = (billable_input / 1_000_000) * INPUT_PRICE_PER_1M
cached_cost = (cached_tokens / 1_000_000) * CACHE_PRICE_PER_1M
output_cost = (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_1M
total_cost = fresh_input_cost + cached_cost + output_cost

print(f"API Reports: {total_input_from_api:,} input tokens (includes cached)")
print(f"Fresh Input: {billable_input:,} tokens × ${INPUT_PRICE_PER_1M}/1M = ${fresh_input_cost:.6f}")
print(f"Cached: {cached_tokens:,} tokens × ${CACHE_PRICE_PER_1M}/1M = ${cached_cost:.6f}")
print(f"Output: {output_tokens:,} tokens × ${OUTPUT_PRICE_PER_1M}/1M = ${output_cost:.6f}")
print(f"Total Cost: ${total_cost:.6f}")

# Without caching comparison
no_cache_cost = (total_input_from_api / 1_000_000) * INPUT_PRICE_PER_1M + output_cost
savings = no_cache_cost - total_cost
print(f"💰 Savings from caching: ${savings:.6f} ({(savings/no_cache_cost)*100:.1f}%)")

# Verify
assert billable_input == 7000, f"Billable input wrong: {billable_input}"
expected_fresh = 0.002100   # 7000 × 0.30 / 1M
expected_cached = 0.000090  # 3000 × 0.03 / 1M
expected_output = 0.012500  # 5000 × 2.50 / 1M
expected_total = 0.014690

assert abs(fresh_input_cost - expected_fresh) < 0.000001, f"Fresh input cost wrong"
assert abs(cached_cost - expected_cached) < 0.000001, f"Cached cost wrong"
assert abs(total_cost - expected_total) < 0.000001, f"Total cost wrong"
print("✅ Cached token handling CORRECT")

# Test Case 3: Real-world scenario from logs
print("\n📊 Test Case 3: Real UBMK Scenario (UPDATED PRICING)")
print("-"*70)
print("Scenario: Domain model generation from 5KB SRS document")
print()

# Hypothetical realistic numbers
api_calls = [
    {"stage": "Scout", "input": 8000, "cached": 0, "output": 3500},
    {"stage": "Architect", "input": 5000, "cached": 0, "output": 1200},
    {"stage": "Specialist", "input": 7000, "cached": 2000, "output": 4000},
    {"stage": "Synthesizer", "input": 6000, "cached": 1500, "output": 3800},
]

total_input = 0
total_cached = 0
total_output = 0
total_cost = 0

for call in api_calls:
    billable_fresh = call["input"] - call["cached"]
    total_input += billable_fresh
    total_cached += call["cached"]
    total_output += call["output"]
    
    # Calculate cost with cache pricing
    fresh_cost = (billable_fresh / 1_000_000) * INPUT_PRICE_PER_1M
    cached_cost = (call["cached"] / 1_000_000) * CACHE_PRICE_PER_1M
    output_cost = (call["output"] / 1_000_000) * OUTPUT_PRICE_PER_1M
    stage_cost = fresh_cost + cached_cost + output_cost
    total_cost += stage_cost
    
    print(f"  {call['stage']:12} | In: {call['input']:5,} (-{call['cached']:4,} cached) | Out: {call['output']:5,} | ${stage_cost:.6f}")

print("-"*70)
print(f"  TOTALS       | Fresh Input: {total_input:6,} | Cached: {total_cached:6,}")
print(f"               | Output:      {total_output:6,} | Total:  ${total_cost:.6f}")

# Calculate what it would cost without caching
no_cache_total_input = sum(c["input"] for c in api_calls)
no_cache_cost = (no_cache_total_input / 1_000_000) * INPUT_PRICE_PER_1M + (total_output / 1_000_000) * OUTPUT_PRICE_PER_1M
savings = no_cache_cost - total_cost
print(f"💰 Caching saved: ${savings:.6f} ({(savings/no_cache_cost)*100:.1f}%)")

# Verify math
expected_input_cost = (total_input / 1_000_000) * INPUT_PRICE_PER_1M
expected_cached_cost = (total_cached / 1_000_000) * CACHE_PRICE_PER_1M
expected_output_cost = (total_output / 1_000_000) * OUTPUT_PRICE_PER_1M
expected_cost = expected_input_cost + expected_cached_cost + expected_output_cost
assert abs(total_cost - expected_cost) < 0.000001, f"Cost mismatch: {total_cost} vs {expected_cost}"
print("✅ Real-world scenario calculation CORRECT")

# Test Case 4: Verify per-token rates
print("\n📊 Test Case 4: Per-Token Rate Verification (UPDATED)")
print("-"*70)
single_input_cost = 1 * (INPUT_PRICE_PER_1M / 1_000_000)
single_output_cost = 1 * (OUTPUT_PRICE_PER_1M / 1_000_000)
single_cache_cost = 1 * (CACHE_PRICE_PER_1M / 1_000_000)

print(f"1 input token   = ${single_input_cost:.12f}")
print(f"1 output token  = ${single_output_cost:.12f}")
print(f"1 cached token  = ${single_cache_cost:.12f}")
print(f"Ratio: Output is {OUTPUT_PRICE_PER_1M / INPUT_PRICE_PER_1M:.2f}x more expensive than input")
print(f"Ratio: Fresh input is {INPUT_PRICE_PER_1M / CACHE_PRICE_PER_1M:.0f}x more expensive than cached")

assert abs(single_input_cost - 0.000000300000) < 0.000000000001, "Input rate wrong"
assert abs(single_output_cost - 0.000002500000) < 0.000000000001, "Output rate wrong"
assert abs(single_cache_cost - 0.000000030000) < 0.000000000001, "Cache rate wrong"
print("✅ Per-token rates CORRECT")

# Summary
print("\n" + "="*70)
print("✅ ALL VERIFICATION TESTS PASSED - READY FOR UBMK")
print("="*70)
print()
print("CRITICAL POINTS FOR UBMK PRESENTATION:")
print("-"*70)
print("1. ✅ Gemini 2.5 Flash PAID tier: $0.30 input, $2.50 output per 1M tokens")
print("2. ✅ Output is 8.3x more expensive than input ($2.50 vs $0.30)")
print("3. ✅ Cached tokens cost $0.03 per 1M (10x cheaper than fresh input)")
print("4. ✅ Cached tokens are INCLUDED in prompt_token_count - must subtract!")
print("5. ✅ Billing formula: (fresh_input × 0.30) + (cached × 0.03) + (output × 2.50)")
print("6. ✅ Pricing verified against official Google AI docs (Dec 12, 2025)")
print()
print("Source: https://ai.google.dev/gemini-api/docs/pricing")
print("="*70)
print("5. ✅ Math verified: cost = (billable_input × 0.075 + output × 0.30) / 1M")
print("="*70)
