"""Code examples from Chapter 08: cost-optimization"""

# --- Example 1 ---
import tiktoken

def count_tokens(text, model="gpt-4o-mini"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def estimate_cost(input_text, output_tokens=500, model="gpt-4o-mini"):
    PRICES = {
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4o": {"input": 0.0025, "output": 0.01},
    }
    
    input_tokens = count_tokens(input_text, model)
    prices = PRICES[model]
    
    return (input_tokens / 1000) * prices["input"] + (output_tokens / 1000) * prices["output"]

# --- Example 2 ---
from enum import Enum

class TaskComplexity(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"

MODEL_MAP = {
    TaskComplexity.SIMPLE: "gpt-4o-mini",    # classification, extraction
    TaskComplexity.MEDIUM: "gpt-4o-mini",    # summaries, moderate extraction
    TaskComplexity.COMPLEX: "gpt-4o",        # reasoning, nuanced analysis
}

# --- Example 3 ---
import re

def remove_boilerplate(text):
    # Kill excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove repeated headers/footers
    patterns = [
        r'Page \d+ of \d+',
        r'Confidential.*\n',
        r'Copyright .*\n',
        r'All rights reserved.*\n',
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text.strip()

# --- Example 4 ---
def smart_truncate(text, max_tokens=2000, model="gpt-4o-mini"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    # 70% from the start, 30% from the end
    keep_start = int(max_tokens * 0.7)
    keep_end = max_tokens - keep_start
    
    truncated = tokens[:keep_start] + tokens[-keep_end:]
    return encoding.decode(truncated)

# --- Example 5 ---
# This generates 200+ tokens of output per call:
"""Please analyze this document and provide a detailed summary of the key 
points, including all entities mentioned, their relationships, and any 
important dates or figures."""

# This generates ~50 tokens:
"""Extract as JSON:
{"entities": [], "dates": [], "key_points": [max 3]}
No explanation."""

# --- Example 6 ---
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    max_tokens=200  # hard ceiling
)

# --- Example 7 ---
import hashlib

class LLMCache:
    def __init__(self, storage_path):
        self.path = Path(storage_path)
        self.path.mkdir(parents=True, exist_ok=True)
    
    def _key(self, prompt, model):
        content = f"{model}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, prompt, model):
        cache_file = self.path / f"{self._key(prompt, model)}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)["response"]
        return None
    
    def set(self, prompt, model, response):
        cache_file = self.path / f"{self._key(prompt, model)}.json"
        with open(cache_file, 'w') as f:
            json.dump({"response": response, "cached_at": datetime.utcnow().isoformat()}, f)

# --- Example 8 ---
class SemanticCache:
    def __init__(self, embedder, threshold=0.95):
        self.embedder = embedder
        self.threshold = threshold
        self.cache = []
    
    def get(self, prompt):
        embedding = self.embedder.embed(prompt)
        for cached_emb, _, response in self.cache:
            if cosine_similarity(embedding, cached_emb) >= self.threshold:
                return response
        return None

# --- Example 9 ---
def batch_classify(documents, batch_size=10):
    results = []
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        
        prompt = "Classify each document:\n\n"
        for j, doc in enumerate(batch):
            prompt += f"[Doc {j+1}]: {doc.content[:500]}...\n\n"
        prompt += "Return JSON array of classifications."
        
        response = llm_client.complete(prompt)
        batch_results = json.loads(response)
        results.extend(batch_results)
    
    return results

# --- Example 10 ---
@dataclass
class UsageRecord:
    date: date
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    task: str

class CostTracker:
    def __init__(self):
        self.records = []
    
    def record(self, model, input_tokens, output_tokens, task):
        cost = self._calculate_cost(model, input_tokens, output_tokens)
        self.records.append(UsageRecord(
            date=date.today(), model=model,
            input_tokens=input_tokens, output_tokens=output_tokens,
            cost=cost, task=task
        ))
    
    def daily_report(self):
        today_records = [r for r in self.records if r.date == date.today()]
        return {
            "total_cost": sum(r.cost for r in today_records),
            "total_tokens": sum(r.input_tokens + r.output_tokens for r in today_records),
            "by_task": self._group_by(today_records, "task"),
        }

