"""Code examples from Chapter 05: prompt-engineering"""

# --- Example 1 ---
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

# --- Example 2 ---
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

# --- Example 3 ---
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

# --- Example 4 ---
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

# --- Example 5 ---
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

# --- Example 6 ---
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

# --- Example 7 ---
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

# --- Example 8 ---
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

# --- Example 9 ---
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

