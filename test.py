from src.models.llm_client import get_llm_client

client = get_llm_client()

resp = client.generate("Say hello in one word")

print(resp.raw_text)
print(resp.latency_ms)