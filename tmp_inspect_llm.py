from src.models.llm_client import get_llm_client, LLMResponse

client = get_llm_client()
print('client type:', type(client).__name__)
resp = client.generate(
    prompt="Rate the Risk Taking facet for: 'I quit my job to start a startup.'",
    system_prompt="Return a JSON object with keys: score (int 1-5), rationale (str), evidence_span (str).",
)
print('is LLMResponse:', isinstance(resp, LLMResponse))
print('raw_text len:', len(resp.raw_text))
print('raw_text repr:', repr(resp.raw_text)[:500])
print('latency_ms:', resp.latency_ms, type(resp.latency_ms))
print('model_name:', resp.model_name, type(resp.model_name))
print('logprobs:', resp.logprobs, type(resp.logprobs))
assert isinstance(resp, LLMResponse)
assert isinstance(resp.raw_text, str)
assert len(resp.raw_text) > 0
assert isinstance(resp.latency_ms, float)
assert resp.latency_ms >= 0
assert isinstance(resp.model_name, str)
assert resp.logprobs is None or isinstance(resp.logprobs, list)
print('all assertions passed')
