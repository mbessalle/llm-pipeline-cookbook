# Chapter 10: Deployment Patterns

You've built the pipeline. It works on your laptop. The tests pass. Now you need to actually run this thing in production, which is a different problem entirely.

I've deployed our pipeline three different ways over the past year. Started with a serverless approach (AWS Lambda), migrated to containers when I hit Lambda's timeout limit, and eventually landed on a queue-based architecture when the volume grew. Each transition taught me something. Mostly that I should have started one step ahead of where I did.

---

## Start Simpler Than You Think

If you're processing fewer than a thousand documents a day and nobody needs results in real-time, a FastAPI app running in a Docker container on a single server is probably enough. I know that's not a sexy answer. But I've watched teams spend two weeks setting up Kubernetes for a pipeline that processes fifty documents a day. That's infrastructure cosplay, not engineering.

Here's what "simple" looks like:

```python
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
```

Dockerfile:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--timeout", "120", "app:app"]
```

That's it. Four workers, 120-second timeout (LLM calls can be slow), behind nginx or a load balancer. This ran our pipeline for the first four months.

---

## When You Outgrow a Single Server

The signal was clear: during peak hours (Monday mornings, everyone uploading their weekend backlog), the queue of background tasks would grow faster than the workers could process them. Response times climbed, background tasks started failing, and I was manually restarting the server every Monday at 10 AM.

Time for a proper queue.

```yaml
# docker-compose.yml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
  
  worker:
    build: .
    command: celery -A worker worker --loglevel=info --concurrency=4
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    deploy:
      replicas: 3
  
  redis:
    image: redis:7-alpine
```

The API accepts documents and drops them on the queue. Workers pull documents off the queue and process them independently. If a worker crashes, the message goes back on the queue and another worker picks it up. If the API goes down, the workers keep processing. They're decoupled, which is the whole point.

```python
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
```

Three worker replicas with concurrency of 4 each gives us 12 parallel document processors. That's enough for our current volume with headroom. When it's not enough, I'll add more replicas -- horizontal scaling is just changing a number.

---

## Serverless: When It Makes Sense

I still use Lambda for some things. Specifically: webhook handlers, scheduled jobs, and anything that runs infrequently enough that paying for idle servers doesn't make sense.

```python
def handler(event, context):
    body = json.loads(event['body'])
    result = process_document(body['document'])
    return {'statusCode': 200, 'body': json.dumps(result)}
```

The timeout limit (15 minutes on Lambda) was what killed it for our main pipeline. Some documents -- especially large PDFs with OCR -- take 20+ minutes to fully process. You can work around this by splitting the work into multiple Lambda invocations, but at that point you're just building a poor man's queue system and you should use an actual queue.

Lambda is great for the API layer though. Accepts requests, validates input, drops messages on SQS, returns a job ID. Scales to zero when nobody's uploading documents. Scales to thousands when everyone is. We pay maybe $3/month for the API layer.

---

## Configuration: Keep It Out of Your Code

Every deployment environment is slightly different. API keys, model selections, rate limits, queue URLs -- all of this should come from environment variables, not hardcoded strings.

```python
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
```

Pydantic settings does the boring work: loads from environment variables, falls back to .env file, validates types, provides defaults. I've used this pattern on every Python project for the last two years and never looked back.

One thing I learned the hard way: don't put API keys in your .env file and commit it. Use a secret manager. Even for personal projects. I once accidentally pushed an .env file with an OpenAI key and someone racked up $200 in charges before I noticed. GitHub has secret scanning now, but don't rely on it.

---

## Feature Flags: Deploy Without Fear

When I change a chunking strategy or swap a model, I don't want to roll it out to all documents at once. Feature flags let you deploy code changes without activating them, then gradually enable them.

```python
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
```

This is deliberately simple. For a team of one, you don't need LaunchDarkly. A dictionary of booleans is fine. The important thing is having the mechanism so you can deploy a risky change on Friday afternoon, leave it disabled, and flip it on Monday morning when you're awake and watching the metrics.

---

## Health Checks: Know When You're Down

Two health check endpoints, minimum:

```python
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
```

`/health` tells you the process is alive. `/ready` tells you it can actually do work. Your load balancer should check `/health` and only route traffic to healthy instances. Your deployment should check `/ready` before marking a new version as live.

The OpenAI health check is just a quick `client.models.list()` call. It adds maybe 200ms to the readiness check, but it catches the scenario where your API key is expired or the OpenAI API is down. Better to fail the readiness check than to accept documents you can't process.

---

## The Deployment Checklist I Actually Use

Before every deployment:
- Tests pass (including the prompt consistency tests from chapter 5)
- Environment variables are set in the target environment
- API keys are in the secret manager, not in code
- Resource limits are configured (memory, CPU, timeout)
- Monitoring dashboards are up and I know what "normal" looks like

During deployment:
- Rolling update, not big-bang replacement
- Watch error rates for the first 10 minutes
- Check latency hasn't spiked
- Verify the queue is draining normally

After deployment:
- Run a smoke test (process one known document, verify output)
- Check costs are in expected range
- Look at the quality metrics from chapter 9

I keep this as a literal checklist in our repo. It takes five minutes to go through and it's caught problems every single time I thought "this change is too small to worry about."

---

## Where to Go From Here

You've got the patterns. Architecture, ingestion, chunking, embeddings, prompts, API resilience, error handling, cost optimization, monitoring, deployment. Ten chapters of stuff I wish someone had told me a year ago.

The best advice I can give: start simple, measure everything, and only add complexity when you have evidence you need it. A well-monitored simple pipeline beats a complex one you can't debug.

And run that cost calculator before you quote anyone a price. Seriously.

Good luck out there.
