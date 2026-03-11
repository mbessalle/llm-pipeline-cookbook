# Chapter 8: Cost Optimization

My first month running LLM pipelines in production, the OpenAI bill was about $340. Not catastrophic, but way higher than my back-of-napkin estimate of $50. I'd been sending entire documents -- headers, footers, page numbers, copyright notices, all of it -- to GPT-4 for every single extraction. And every response was a verbose paragraph when all I needed was a JSON object.

After two weeks of optimization, I got the same pipeline down to about $45/month. Same documents, same output quality. The techniques aren't complicated, they just require actually thinking about what you're sending and receiving.

---

## The Simple Math

```
Cost = (Input Tokens x Input Price) + (Output Tokens x Output Price)
```

Four levers to pull: reduce input tokens, reduce output tokens, use cheaper models, or make fewer calls. That's it. Everything in this chapter maps to one of those four.

---

## Count Before You Optimize

You can't optimize what you don't measure. Before changing anything, I added token counting to every API call:

```python
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
```

I ran this on a sample of 100 documents and discovered that our average input was 3,200 tokens per call, of which about 800 were boilerplate that appeared on every single page. Nearly 25% of our input cost was paying the LLM to read "Page 3 of 47 -- Confidential" over and over.

---

## Use the Right Model for the Job

This is the single biggest cost saver. Not every task needs GPT-4.

Document classification? GPT-4o-mini handles it fine. Entity extraction from well-structured text? Mini. Date parsing? Mini. You only need the expensive model for tasks that actually require reasoning -- analyzing ambiguous contract clauses, understanding complex regulatory language, making judgment calls on edge cases.

```python
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
```

Here's what this looked like for us on 10,000 documents:

Sending everything to GPT-4o: roughly $300/month. Using GPT-4o-mini for the 85% of tasks that don't need reasoning: about $50. That's the same output quality -- I verified it on a test set of 200 labeled documents. Classification accuracy dropped by 0.3%. Summary quality was indistinguishable. The only place GPT-4o made a meaningful difference was on complex contract analysis.

---

## Stop Sending Garbage to the API

The amount of useless text I was sending to the API before I started cleaning inputs was embarrassing.

```python
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
```

For our municipal documents, this alone cut input tokens by 20%. Those page numbers and repeated headers on every page? Gone. The three blank lines between paragraphs? Collapsed to one.

When you need to truncate, be smart about it. Don't just cut off at a character limit. Keep the beginning (usually the most important context) and the end (often contains conclusions or summaries):

```python
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
```

---

## Make the LLM Shut Up

Sounds harsh, but your prompts might be encouraging verbose output when you only need structured data.

```python
# This generates 200+ tokens of output per call:
"""Please analyze this document and provide a detailed summary of the key 
points, including all entities mentioned, their relationships, and any 
important dates or figures."""

# This generates ~50 tokens:
"""Extract as JSON:
{"entities": [], "dates": [], "key_points": [max 3]}
No explanation."""
```

That's a 4x reduction in output tokens. At 10,000 calls per month with GPT-4o-mini, the verbose prompt costs about $1.20 in output tokens. The concise prompt costs $0.30. Small individually, but it adds up across every prompt in your pipeline.

And always set max_tokens:

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    max_tokens=200  # hard ceiling
)
```

Without this, a confused model can generate thousands of tokens of irrelevant output. Ask me how I know.

---

## Cache Everything

If the same document comes through twice -- and it will -- don't pay for it twice.

```python
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
```

Our cache hit rate is about 12%. Not massive, but that's 12% of API calls we're not paying for. Documents get resubmitted more often than you'd think -- updated versions, duplicate uploads, reprocessing after pipeline changes.

For an even bigger win, semantic caching matches documents that are similar but not identical:

```python
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
```

Set the threshold high (0.95+). You don't want false cache hits returning results for the wrong document.

---

## Batch Calls When Possible

Instead of one API call per document for classification, batch ten documents into a single call:

```python
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
```

Fewer API calls means less overhead per request (system prompts aren't repeated for each document) and more efficient use of the context window. For classification, batching 10 documents into one call cut our costs for that stage by about 60%.

The catch: if one document in the batch causes an error, you might lose the whole batch. I keep batch sizes to 10 or fewer so the blast radius of a failure is small.

---

## Track Your Spending

I built a simple cost tracker that logs every API call. Nothing fancy, but the daily report has caught cost spikes before they became cost disasters.

```python
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
```

We review costs weekly. The report breaks down spending by task, so we can see exactly which pipeline stage is the most expensive. Last month, we discovered that our "summary" task was costing 3x what we expected because a prompt change had accidentally removed the max_tokens constraint. Found it in the cost report, fixed it in ten minutes, saved about $80/month.

---

*Next: your pipeline is cheap and reliable. But is it actually working well? Monitoring and observability are how you know.*
