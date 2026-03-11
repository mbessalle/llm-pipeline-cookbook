"""Code examples from Chapter 10: deployment"""

# --- Example 1 ---
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

app = FastAPI()

class ProcessRequest(BaseModel):
    document: str
    callback_url: str | None = None

@app.post("/process")
async def process_document(request: ProcessRequest, background: BackgroundTasks):
    job_id = generate_job_id()
    background.add_task(process_and_store, job_id, request.document, request.callback_url)
    return {"job_id": job_id, "status": "processing"}

@app.get("/jobs/{job_id}")
async def get_status(job_id: str):
    return get_job(job_id)

# --- Example 2 ---
from celery import Celery

app = Celery('pipeline', broker='redis://localhost:6379/0')

@app.task(bind=True, max_retries=3)
def process_document_task(self, document_id, document_text):
    try:
        result = process_document(document_text)
        store_result(document_id, result)
        return {"status": "success", "document_id": document_id}
    except Exception as e:
        # Exponential backoff on retry
        self.retry(exc=e, countdown=60 * (2 ** self.request.retries))

# --- Example 3 ---
def handler(event, context):
    body = json.loads(event['body'])
    result = process_document(body['document'])
    return {'statusCode': 200, 'body': json.dumps(result)}

# --- Example 4 ---
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    default_model: str = "gpt-4o-mini"
    max_tokens: int = 2000
    requests_per_minute: int = 60
    redis_url: str = "redis://localhost:6379"
    
    class Config:
        env_file = ".env"

settings = Settings()

# --- Example 5 ---
from enum import Enum

class Feature(Enum):
    USE_GPT4 = "use_gpt4"
    NEW_CHUNKING = "new_chunking"
    ENABLE_CACHING = "enable_caching"

class FeatureFlags:
    def __init__(self):
        self._flags = {
            Feature.USE_GPT4: False,
            Feature.NEW_CHUNKING: False,
            Feature.ENABLE_CACHING: True,
        }
    
    def enabled(self, feature):
        return self._flags.get(feature, False)

flags = FeatureFlags()

model = "gpt-4o" if flags.enabled(Feature.USE_GPT4) else "gpt-4o-mini"

# --- Example 6 ---
@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/ready")
async def ready():
    checks = {
        "database": check_database(),
        "redis": check_redis(),
        "openai": check_openai(),
    }
    return {"ready": all(checks.values()), "checks": checks}

