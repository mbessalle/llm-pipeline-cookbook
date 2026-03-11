# Chapter 5: Prompt Engineering for Pipelines

Forget everything you know about prompt engineering from ChatGPT. Seriously. Chat prompting is about creativity, flexibility, getting interesting responses. Pipeline prompting is the opposite. You want boring. You want predictable. You want the LLM to return the exact same structure on the ten thousandth document that it returned on the first.

I learned this the hard way. My first pipeline prompts were conversational, flexible, "creative." And they worked great -- on the five test documents I tried. Then I ran them on 500 real documents and got back a beautiful variety of output formats, none of which my parser could handle.

---

## Always. Use. Structured. Output.

Never, under any circumstances, ask an LLM to return free text that you then try to parse with regex. I know it's tempting. It seems simpler. It will break at 2 AM on a Sunday when a document has an unexpected format and your regex matches the wrong thing.

### JSON Mode (Minimum Acceptable)

```python
from openai import OpenAI

client = OpenAI()

def extract_entities(text: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": """Extract entities from the document.
                
Return JSON with this exact structure:
{
    "people": [{"name": str, "role": str}],
    "organizations": [{"name": str, "type": str}],
    "dates": [{"date": str, "context": str}],
    "amounts": [{"value": str, "currency": str, "context": str}]
}

If a field is not found, use an empty array."""
            },
            {"role": "user", "content": text}
        ]
    )
    
    return json.loads(response.choices[0].message.content)
```

### Function Calling (What I Actually Use)

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    role: str | None = None

class Organization(BaseModel):
    name: str
    type: str | None = None

class ExtractedEntities(BaseModel):
    people: list[Person]
    organizations: list[Organization]
    dates: list[str]

def extract_with_functions(text: str) -> ExtractedEntities:
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract entities from the document."},
            {"role": "user", "content": text}
        ],
        response_format=ExtractedEntities
    )
    
    return response.choices[0].message.parsed
```

Function calling beats JSON mode because the schema is enforced at the API level, you get Pydantic validation for free, and your IDE gives you autocomplete on the results. After switching, our parsing failures dropped from about 3% to essentially zero.

---

## Show the Model What You Want

Few-shot prompting -- giving examples of input/output pairs -- is the single most effective technique for pipeline prompts. Long instructions about what to do are ambiguous. Examples are not.

```python
FEW_SHOT_TEMPLATE = """Classify the document into one of these categories:
- CONTRACT: Legal agreements, terms of service
- INVOICE: Bills, payment requests
- REPORT: Analysis, summaries, findings
- CORRESPONDENCE: Letters, emails, memos
- OTHER: None of the above

Examples:

Document: "This Agreement is entered into as of January 1, 2024, between Party A and Party B..."
Classification: CONTRACT
Reasoning: Contains "Agreement", mentions parties, has formal legal language.

Document: "Invoice #12345. Amount Due: $5,000. Payment due within 30 days..."
Classification: INVOICE
Reasoning: Contains invoice number, amount due, payment terms.

Document: "Dear Mr. Smith, Thank you for your inquiry regarding..."
Classification: CORRESPONDENCE
Reasoning: Letter format, greeting, personal address.

Now classify this document:

Document: {document_text}
Classification:"""
```

Two to four examples is usually the sweet spot. More than that and you're burning tokens without much accuracy gain. Include at least one edge case -- a document that's borderline between categories, or one with unusual formatting. The model generalizes from your examples, so make them representative of the mess you'll encounter in production.

---

## Chain-of-Thought for the Hard Stuff

Some extraction tasks need the model to reason through multiple steps. Contract analysis is a good example -- you need to identify parties, then find dates, then connect financial terms to the right parties. If you just ask for all of it at once, the model skips steps and makes mistakes.

```python
COT_PROMPT = """Analyze this contract and extract key terms.

Think through this step by step:

1. First, identify the parties involved
2. Then, find the effective date and term length
3. Next, locate any payment terms or amounts
4. Finally, note any termination clauses

Document:
{document}

Analysis:
Step 1 - Parties:
[identify all parties]

Step 2 - Dates:
[find dates and duration]

Step 3 - Financial:
[extract payment details]

Step 4 - Termination:
[note termination conditions]

Final Extraction (JSON):
"""
```

This is slower and more expensive per call -- more output tokens means more cost. But for complex documents where accuracy matters, the accuracy improvement is worth it. We use chain-of-thought for contract analysis and financial document extraction. For simple classification? No need, just adds cost.

---

## Handle the Edge Cases in the Prompt

Documents fail in predictable ways. Missing fields, unexpected languages, corrupted text. If you don't tell the model what to do in these cases, it'll improvise, and you won't like the result.

```python
ROBUST_PROMPT = """Extract information from this document.

IMPORTANT:
- If a field is not found, use null (not empty string, not "N/A")
- If a field is ambiguous, include all possibilities in an array
- If the document is not in English, still extract what you can
- If the document appears corrupted or unreadable, return {"error": "unreadable"}

Fields to extract:
- title: document title (null if not found)
- date: ISO format (null if not found)
- author: name (null if not found)
- summary: 1-2 sentence summary (always provide this)

Document:
{document}

Return JSON only."""
```

That `null` instruction is critical. Without it, you'll get a mix of empty strings, "N/A", "not found", "none", "unknown", and a dozen other variations across your ten thousand documents. Your downstream code needs to handle one null pattern, not twelve string patterns.

---

## Version Your Prompts Like Code

When I first started, prompts lived as string constants scattered across the codebase. One day I changed a prompt to fix an edge case and broke three other things. Now we version them.

```python
PROMPTS = {
    "extraction": {
        "v1": EXTRACTION_V1,   # original
        "v2": EXTRACTION_V2,   # added null handling
        "v3": EXTRACTION_V3,   # fixed Dutch document support
        "latest": "v3"
    }
}

def get_prompt(name: str, version: str = "latest") -> str:
    versions = PROMPTS[name]
    if version == "latest":
        version = versions["latest"]
    return versions[version]
```

When I change a prompt, the old version stays. I can A/B test the new one against the old one on the same documents. If the new version is worse on some document types, I can roll back in one line. It's like database migrations but for prompts.

---

## Test Your Prompts

This seems obvious but almost nobody does it. Prompts are code. Test them.

```python
import pytest

TEST_CASES = [
    {
        "input": "Agreement between Company A and Company B dated January 1, 2024",
        "expected": {"classification": "CONTRACT"}
    },
    {
        "input": "Invoice #123 - $500 due",
        "expected": {"classification": "INVOICE"}
    }
]

@pytest.mark.parametrize("case", TEST_CASES)
def test_classification(case):
    result = classify_document(case["input"])
    assert result["classification"] == case["expected"]["classification"]

def test_consistency():
    """Same input, same output. Five times."""
    text = "Standard contract text..."
    results = [classify_document(text) for _ in range(5)]
    classifications = [r["classification"] for r in results]
    assert len(set(classifications)) == 1
```

That consistency test has caught problems I would have otherwise missed. Temperature 0 helps, but even at temperature 0, LLMs aren't perfectly deterministic. If the same document gets classified differently on different runs, your prompt is ambiguous and needs to be tighter.

We run these tests on every prompt change as part of CI. Each test costs maybe $0.01 in API calls. Finding a broken prompt in production costs a lot more.

---

## Save Tokens, Save Money

A prompt that runs once? Who cares about length. A prompt that runs ten thousand times a month? Every token counts.

```python
# This costs ~80 tokens per call in system message alone:
"""I would like you to please analyze the following document and extract 
some key information from it. The information I'm looking for includes
the names of any people mentioned, any organizations that are referenced,
and any dates that appear in the text. Please format your response as JSON."""

# This does the same thing in ~30 tokens:
"""Extract from document:
- people: [{name, role}]
- orgs: [{name, type}]  
- dates: [ISO format]

Return JSON only."""
```

Fifty tokens saved per call. At 10,000 calls a month with gpt-4o-mini, that's about $0.075 saved. Not life-changing. But with gpt-4o, it's $1.50. And it adds up across every prompt in your pipeline.

The bigger wins come from truncating input intelligently. Don't send a 20-page document when you only need the first three pages. Don't include boilerplate headers and footers that appear on every page. Trim what you know is irrelevant before it hits the API.

---

## A Simple Template System

Nothing fancy. Just enough structure to keep things organized.

```python
from string import Template
from enum import Enum

class PromptType(Enum):
    CLASSIFY = "classify"
    EXTRACT = "extract"
    SUMMARIZE = "summarize"
    QA = "qa"

class PromptLibrary:
    _templates = {
        PromptType.CLASSIFY: Template(
            "Classify into: $categories\n\nDocument: $document\n\n"
            'Return: {"category": "...", "confidence": 0.0-1.0}'
        ),
        PromptType.SUMMARIZE: Template(
            "Summarize in $max_sentences sentences.\n\n"
            "Document: $document\n\nSummary:"
        ),
        PromptType.QA: Template(
            "Context: $context\n\nQuestion: $question\n\n"
            'Answer based only on the context. If not found, say "Not found in document."\n\n'
            "Answer:"
        ),
    }
    
    @classmethod
    def get(cls, prompt_type: PromptType, **kwargs) -> str:
        return cls._templates[prompt_type].safe_substitute(**kwargs)
```

I considered using Jinja2 or something more powerful. Decided against it. The simpler the template system, the easier it is to debug when something goes wrong at 2 AM. And something always goes wrong at 2 AM.

---

*Next: your prompts are great, but the API will let you down. Rate limits, timeouts, random errors -- let's build resilient API clients.*
