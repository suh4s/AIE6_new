# Detailed Plan: Implementing Semantic LLM Caching

**Version:** 1.0
**Date:** October 26, 2023

## Table of Contents

1.  [Introduction & Goals](#1-introduction--goals)
    1.1. [Problem Statement](#11-problem-statement)
    1.2. [Proposed Solution: Semantic LLM Caching](#12-proposed-solution-semantic-llm-caching)
    1.3. [Objectives](#13-objectives)
    1.4. [Scope](#14-scope)
2.  [Technology Stack](#2-technology-stack)
3.  [Project Setup & Environment](#3-project-setup--environment)
    3.1. [Directory Structure](#31-directory-structure)
    3.2. [Virtual Environment](#32-virtual-environment)
    3.3. [Initial Dependencies (`requirements.txt`)](#33-initial-dependencies-requirementstxt)
    3.4. [Configuration Management](#34-configuration-management)
4.  [Phase 1: Core Components & Services](#4-phase-1-core-components--services)
    4.1. [Embedding Service](#41-embedding-service)
    4.2. [LLM Service](#42-llm-service)
    4.3. [Redis (Cache) Service Setup](#43-redis-cache-service-setup)
        4.3.1. [Data Schema in Redis](#431-data-schema-in-redis)
        4.3.2. [RediSearch Index Definition](#432-redisearch-index-definition)
5.  [Phase 2: Caching Logic Implementation](#5-phase-2-caching-logic-implementation)
    5.1. [Redis Connection Manager](#51-redis-connection-manager)
    5.2. [Cache Interaction Module (`cache_manager.py`)](#52-cache-interaction-module-cache_managerpy)
        5.2.1. [Function: `get_cached_response`](#521-function-get_cached_response)
        5.2.2. [Function: `set_cached_response`](#522-function-set_cached_response)
6.  [Phase 3: API Development (FastAPI)](#6-phase-3-api-development-fastapi)
    6.1. [Main Application File (`main.py`)](#61-main-application-file-mainpy)
    6.2. [Request/Response Models (Pydantic)](#62-requestresponse-models-pydantic)
    6.3. [API Endpoint: `/chat_semantic_cache`](#63-api-endpoint-chat_semantic_cache)
7.  [Phase 4: Dockerization](#7-phase-4-dockerization)
    7.1. [`Dockerfile` for the Application](#71-dockerfile-for-the-application)
    7.2. [`docker-compose.yml`](#72-docker-composeyml)
8.  [Phase 5: Testing Strategy](#8-phase-5-testing-strategy)
    8.1. [Unit Tests](#81-unit-tests)
    8.2. [Integration Tests](#82-integration-tests)
    8.3. [Performance Tests (Cache Hit/Miss)](#83-performance-tests-cache-hitmiss)
9.  [Phase 6: Deployment (Local) & Usage](#9-phase-6-deployment-local--usage)
    9.1. [Running with Docker Compose](#91-running-with-docker-compose)
    9.2. [Example API Calls](#92-example-api-calls)
10. [Phase 7: Documentation](#10-phase-7-documentation)
    10.1. [README.md](#101-readmemd)
    10.2. [Code Comments](#102-code-comments)
11. [Future Enhancements & Considerations](#11-future-enhancements--considerations)

---

## 1. Introduction & Goals

### 1.1. Problem Statement
Standard LLM caching mechanisms often rely on exact-match string comparisons for prompts. This is inefficient as semantically similar prompts (e.g., "What's the weather like?" vs. "Tell me about the current weather.") are treated as distinct, leading to redundant LLM calls, increased latency, and higher operational costs. In-memory caches are also volatile and not suitable for production or distributed systems.

### 1.2. Proposed Solution: Semantic LLM Caching
This project aims to implement a more intelligent caching system that:
*   Utilizes a persistent database (Redis Stack) for storing cache entries.
*   Employs semantic similarity to determine cache hits. Prompts are converted to vector embeddings, and a vector search is performed in the cache. If a sufficiently similar prompt embedding is found, its corresponding LLM response is served.

### 1.3. Objectives
*   Develop a locally runnable application demonstrating semantic LLM caching.
*   Reduce redundant LLM API calls for semantically similar queries.
*   Improve response latency for cached queries.
*   Use a database (Redis) for persistent and scalable caching.
*   Containerize the application using Docker for ease of deployment and reproducibility.

### 1.4. Scope
*   **In Scope:**
    *   Semantic caching for LLM prompt-response pairs.
    *   Using Redis Stack for vector storage and search.
    *   A Python-based application (FastAPI).
    *   Dockerization for local deployment.
    *   Basic API endpoint for interaction.
*   **Out of Scope:**
    *   Production-grade deployment to cloud platforms.
    *   Advanced UI/UX.
    *   Complex cache invalidation strategies beyond simple TTL (Time-To-Live), if implemented.
    *   User authentication/authorization for the API.
    *   End-to-End (E2E) caching of entire RAG pipelines (focus is on LLM prompt caching).

---

## 2. Technology Stack
*   **Programming Language:** Python 3.9+
*   **Web Framework:** FastAPI
*   **Database/Cache:** Redis Stack (specifically leveraging RediSearch for vector similarity search)
*   **Embedding Model:** A sentence-transformer model (e.g., `all-MiniLM-L6-v2` for efficiency) or a Hugging Face Inference Endpoint.
*   **LLM:** Hugging Face Inference Endpoint or a local LLM (if feasible for local setup).
*   **Containerization:** Docker, Docker Compose
*   **Python Libraries:**
    *   `fastapi`
    *   `uvicorn`
    *   `pydantic` (for data validation)
    *   `redis` (Python client for Redis)
    *   `sentence-transformers` (if using local embeddings)
    *   `langchain` (or `langchain-huggingface`, `langchain-community` for LLM and embedding integration)
    *   `python-dotenv` (for managing environment variables)
    *   `numpy` (for vector operations)

---

## 3. Project Setup & Environment

### 3.1. Directory Structure
```
/semantic-llm-cache-app
|-- app/
|   |-- __init__.py
|   |-- main.py             # FastAPI application
|   |-- core/
|   |   |-- __init__.py
|   |   |-- config.py       # Configuration settings
|   |   |-- security.py     # API keys, etc. (if added)
|   |-- services/
|   |   |-- __init__.py
|   |   |-- embedding_service.py
|   |   |-- llm_service.py
|   |-- cache/
|   |   |-- __init__.py
|   |   |-- redis_connector.py # Manages Redis connection and index setup
|   |   |-- cache_manager.py   # Logic for get/set cache
|   |-- models/
|   |   |-- __init__.py
|   |   |-- pydantic_models.py # Pydantic request/response models
|-- tests/
|   |-- __init__.py
|   |-- test_cache_manager.py
|   |-- test_api.py
|-- .env                    # Environment variables (API keys, Redis URL)
|-- .gitignore
|-- Dockerfile              # For the FastAPI application
|-- docker-compose.yml      # For running app and Redis
|-- requirements.txt
|-- README.md
```

### 3.2. Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\\Scripts\\activate  # Windows
pip install --upgrade pip
```

### 3.3. Initial Dependencies (`requirements.txt`)
```
fastapi
uvicorn[standard]
pydantic
redis
sentence-transformers  # Or specific langchain packages for HF embeddings
langchain             # Or specific langchain packages
langchain-huggingface
langchain-community
numpy
python-dotenv
```
Install with: `pip install -r requirements.txt`

### 3.4. Configuration Management (`app/core/config.py`)
Use Pydantic's `BaseSettings` to load configurations from environment variables (defined in `.env`).
```python
# app/core/config.py
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    APP_NAME: str = "Semantic LLM Cache API"
    LOG_LEVEL: str = "INFO"

    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str | None = None # if your redis is password protected
    REDIS_INDEX_NAME: str = "prompt_embeddings_idx"
    
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2" # Example for sentence-transformers
    # OR for HF Inference Endpoint for Embeddings
    # HF_EMBEDDING_ENDPOINT_URL: str
    # HF_TOKEN: str # For authenticated HF endpoints

    # LLM_ENDPOINT_URL: str # For HF Inference Endpoint for LLM

    SEMANTIC_SIMILARITY_THRESHOLD: float = 0.90 # Cosine similarity threshold
    CACHE_TTL_SECONDS: int = 3600 # 1 hour

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache() # Cache the settings object
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
```
Create a `.env` file in the root:
```env
# .env
REDIS_HOST=redis_cache_service # Matches docker-compose service name
REDIS_PORT=6379
# HF_EMBEDDING_ENDPOINT_URL=your_embedding_endpoint
# LLM_ENDPOINT_URL=your_llm_endpoint
# HF_TOKEN=your_hf_token
```

---

## 4. Phase 1: Core Components & Services

### 4.1. Embedding Service (`app/services/embedding_service.py`)
Responsible for generating vector embeddings for input text.
```python
# app/services/embedding_service.py
from sentence_transformers import SentenceTransformer # if using local model
# from langchain_huggingface import HuggingFaceEndpointEmbeddings # if using HF endpoint
from app.core.config import settings
import numpy as np
from typing import List

class EmbeddingService:
    def __init__(self):
        # Option 1: Local Sentence Transformer model
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        # Option 2: Hugging Face Inference Endpoint
        # self.model = HuggingFaceEndpointEmbeddings(
        #     model=settings.HF_EMBEDDING_ENDPOINT_URL,
        #     huggingfacehub_api_token=settings.HF_TOKEN
        # )
        print(f"Embedding service initialized with model: {settings.EMBEDDING_MODEL_NAME}")

    def get_embedding(self, text: str) -> List[float]:
        # For SentenceTransformer
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
        # For HuggingFaceEndpointEmbeddings
        # embedding = self.model.embed_query(text)
        # return embedding

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        # For SentenceTransformer
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
        # For HuggingFaceEndpointEmbeddings
        # embeddings = self.model.embed_documents(texts)
        # return embeddings

embedding_service = EmbeddingService() # Singleton instance
```

### 4.2. LLM Service (`app/services/llm_service.py`)
Responsible for interacting with the LLM.
```python
# app/services/llm_service.py
from langchain_huggingface import HuggingFaceEndpoint
from app.core.config import settings
import os

class LLMService:
    def __init__(self):
        # Ensure HF_TOKEN is available if using an authenticated HF endpoint
        hf_token = os.getenv("HF_TOKEN") or settings.HF_TOKEN # Fallback to settings if needed
        # if not hf_token or not settings.LLM_ENDPOINT_URL:
        #     raise ValueError("HF_TOKEN and LLM_ENDPOINT_URL must be set for LLMService")
        
        # self.llm = HuggingFaceEndpoint(
        #     endpoint_url=settings.LLM_ENDPOINT_URL,
        #     huggingfacehub_api_token=hf_token,
        #     task="text-generation", # or the appropriate task
        #     # Add other LLM parameters as needed (e.g., max_new_tokens)
        # )
        # print(f"LLM service initialized with endpoint: {settings.LLM_ENDPOINT_URL}")
        print("LLM Service initialized (ensure endpoint and token are configured if using actual LLM)")


    async def generate_response(self, prompt: str) -> str:
        # try:
        #     response = await self.llm.ainvoke(prompt)
        #     return response
        # except Exception as e:
        #     print(f"Error calling LLM: {e}")
        #     return "Error: Could not get response from LLM."
        print(f"LLMService.generate_response called with prompt: {prompt[:50]}...") # Placeholder
        return f"This is a placeholder LLM response for the prompt: '{prompt[:30]}...'" # Placeholder

llm_service = LLMService() # Singleton instance
```

### 4.3. Redis (Cache) Service Setup

#### 4.3.1. Data Schema in Redis
Each cache entry will be a Redis Hash. We need to store:
*   `prompt_text`: The original prompt string (for readability/debugging, optional for core logic).
*   `prompt_embedding`: The vector embedding of the prompt (as bytes).
*   `llm_response`: The response from the LLM for this prompt.
*   `timestamp`: When the entry was cached (for potential TTL or LRU).
*   `hits`: Number of times this cache entry was served (for analytics).

#### 4.3.2. RediSearch Index Definition (`app/cache/redis_connector.py`)
The `RedisConnector` will handle creating the index if it doesn't exist.
```python
# app/cache/redis_connector.py
import redis
from redis.commands.search.field import VectorField, TextField, NumericField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
import numpy as np
from app.core.config import settings

# Assuming embedding dimension is known (e.g., 384 for all-MiniLM-L6-v2)
EMBEDDING_DIM = 384 # This needs to match your embedding model

class RedisConnector:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD, # None if no password
            decode_responses=False # Store embeddings as bytes
        )
        try:
            self.redis_client.ping()
            print("Successfully connected to Redis.")
            self.create_index_if_not_exists()
        except redis.exceptions.ConnectionError as e:
            print(f"Could not connect to Redis: {e}")
            # Handle connection error appropriately, maybe raise it
            # For now, allow app to start but cache will fail
            self.redis_client = None


    def get_client(self):
        return self.redis_client

    def create_index_if_not_exists(self):
        if not self.redis_client:
            print("Redis client not available. Skipping index creation.")
            return
            
        index_name = settings.REDIS_INDEX_NAME
        try:
            # Check if index exists
            self.redis_client.ft(index_name).info()
            print(f"Index '{index_name}' already exists.")
        except redis.exceptions.ResponseError as e:
            # Index does not exist, create it
            if "Unknown Index name" in str(e):
                print(f"Index '{index_name}' does not exist. Creating index...")
                schema = (
                    TextField("$.prompt_text", as_name="prompt_text"),
                    VectorField(
                        "$.prompt_embedding",
                        "FLAT", # or HNSW for larger datasets
                        {
                            "TYPE": "FLOAT32",
                            "DIM": EMBEDDING_DIM,
                            "DISTANCE_METRIC": "COSINE",
                        },
                        as_name="prompt_embedding",
                    ),
                    TextField("$.llm_response", as_name="llm_response"),
                    NumericField("$.timestamp", as_name="timestamp"),
                    NumericField("$.hits", as_name="hits")
                )
                definition = IndexDefinition(prefix=["prompt_cache:"], index_type=IndexType.JSON)
                self.redis_client.ft(index_name).create_index(fields=schema, definition=definition)
                print(f"Index '{index_name}' created successfully.")
            else:
                # Some other Redis error occurred
                print(f"Redis error when checking/creating index: {e}")
                raise

redis_connector = RedisConnector()
```

---

## 5. Phase 2: Caching Logic Implementation

### 5.1. Redis Connection Manager
Already covered in `app/cache/redis_connector.py`.

### 5.2. Cache Interaction Module (`app/cache/cache_manager.py`)
This module will contain functions to get and set items in the semantic cache.

```python
# app/cache/cache_manager.py
import json
import time
import numpy as np
from redis.commands.search.query import Query
from app.cache.redis_connector import redis_connector, EMBEDDING_DIM
from app.core.config import settings
from app.services.embedding_service import embedding_service # For generating embeddings
from typing import Dict, Any, Optional, Tuple

class SemanticCacheManager:
    def __init__(self):
        self.redis_client = redis_connector.get_client()
        self.index_name = settings.REDIS_INDEX_NAME
        self.embedding_service = embedding_service
        self.similarity_threshold = settings.SEMANTIC_SIMILARITY_THRESHOLD
        self.ttl_seconds = settings.CACHE_TTL_SECONDS

    def _serialize_embedding(self, embedding: list[float]) -> bytes:
        return np.array(embedding, dtype=np.float32).tobytes()

    def _deserialize_embedding(self, embedding_bytes: bytes) -> list[float]:
        return np.frombuffer(embedding_bytes, dtype=np.float32).tolist()

    async def get_cached_response(self, prompt: str) -> Optional[str]:
        if not self.redis_client:
            print("CacheManager: Redis client not available.")
            return None

        prompt_embedding = self.embedding_service.get_embedding(prompt)
        prompt_embedding_bytes = self._serialize_embedding(prompt_embedding)

        # KNN Query
        # Return top 1, filter by vector score (1 - cosine_similarity)
        # So score < (1 - threshold)
        # Note: query_params uses $param_name syntax
        # Ensure the field name in the query matches the 'as_name' in index schema ('prompt_embedding')
        query = (
            Query(f"@prompt_embedding:[VECTOR_RANGE $radius $vec]=>{{$yield_distance_as: vector_score}}")
            .sort_by("vector_score")
            .return_fields("prompt_text", "llm_response", "vector_score", "id") # id is the redis key
            .paging(0, 1) # Get top 1
            .dialect(2)
        )
        
        query_params = {
            "vec": prompt_embedding_bytes,
            "radius": 1 - self.similarity_threshold # similarity > threshold => distance < 1 - threshold
        }

        try:
            results = self.redis_client.ft(self.index_name).search(query, query_params)
            
            if results.total > 0:
                doc = results.docs[0]
                # Check if score is within threshold
                # RediSearch returns distance, so lower is better.
                # vector_score should be <= (1 - similarity_threshold)
                # But Vector_Range uses radius, so this is already filtered. We can double check if needed.
                # For simplicity, we assume if result is returned by VECTOR_RANGE, it's within "radius"
                
                print(f"Semantic cache HIT for prompt: '{prompt[:50]}...' (Similarity: {1-float(doc.vector_score) :.4f}) Key: {doc.id}")
                # Increment hits (optional, can be done async or batched)
                self.redis_client.json().numincrby(doc.id, "$.hits", 1)
                # Update TTL (refresh expiry on access)
                self.redis_client.expire(doc.id, self.ttl_seconds)
                
                # The document fields are JSON strings, need to parse them
                return json.loads(doc.llm_response) # Assuming llm_response was stored as JSON string
            else:
                print(f"Semantic cache MISS for prompt: '{prompt[:50]}...'")
                return None
        except Exception as e:
            print(f"Error querying Redis cache: {e}")
            return None

    async def set_cached_response(self, prompt: str, llm_response: str) -> None:
        if not self.redis_client:
            print("CacheManager: Redis client not available. Cannot set cache.")
            return

        prompt_embedding = self.embedding_service.get_embedding(prompt)
        
        cache_key = f"prompt_cache:{time.time_ns()}" # Unique key, or could be hash of prompt
        
        cache_data = {
            "prompt_text": json.dumps(prompt), # Store as JSON string
            "prompt_embedding": prompt_embedding, # Will be handled by redis client as bytes
            "llm_response": json.dumps(llm_response), # Store as JSON string
            "timestamp": int(time.time()),
            "hits": 0
        }
        
        # Note: redis-py client for JSON handles dict to JSON conversion automatically
        # For embedding, we need to ensure it's passed in a way RediSearch understands (list of floats for JSON, or bytes for direct HSET)
        # The JSON path '$.prompt_embedding' in the schema means we expect the embedding to be part of the JSON document.
        
        try:
            self.redis_client.json().set(cache_key, "$", cache_data)
            self.redis_client.expire(cache_key, self.ttl_seconds)
            print(f"Cached response for prompt: '{prompt[:50]}...' with key {cache_key}")
        except Exception as e:
            print(f"Error setting Redis cache for key {cache_key}: {e}")

semantic_cache_manager = SemanticCacheManager()
```
**Note on `$.prompt_embedding` for JSON index:** When using RediSearch with JSON documents, vector fields are typically expected to be arrays of numbers within the JSON. The `redis-py` client handles the conversion of Python lists to this format when using `json().set()`.

---

## 6. Phase 3: API Development (FastAPI)

### 6.1. Main Application File (`app/main.py`)
```python
# app/main.py
from fastapi import FastAPI, HTTPException, Depends
from app.models.pydantic_models import ChatRequest, ChatResponse
from app.cache.cache_manager import semantic_cache_manager
from app.services.llm_service import llm_service
from app.core.config import Settings, get_settings

app = FastAPI(title=get_settings().APP_NAME)

@app.on_event("startup")
async def startup_event():
    # This is to ensure redis_connector initializes and potentially creates index
    # Accessing it should trigger its __init__ if not already done.
    if semantic_cache_manager.redis_client is None:
        print("WARNING: Redis client is not available on startup. Caching will not work.")
    print("Application startup complete.")


@app.post("/chat_semantic_cache", response_model=ChatResponse)
async def chat_with_semantic_cache(request: ChatRequest, settings: Settings = Depends(get_settings)):
    prompt = request.prompt

    # 1. Try to get response from semantic cache
    cached_response_content = await semantic_cache_manager.get_cached_response(prompt)

    if cached_response_content is not None:
        return ChatResponse(prompt=prompt, response=cached_response_content, source="cache")

    # 2. If cache miss, get response from LLM
    print(f"Cache miss for: '{prompt[:50]}...'. Querying LLM.")
    llm_response_content = await llm_service.generate_response(prompt)

    if "Error:" in llm_response_content: # Basic error check
        raise HTTPException(status_code=500, detail=llm_response_content)

    # 3. Store the new response in cache (fire and forget, or await if critical)
    # For simplicity, running it in the background (not awaiting)
    # In a real app, consider a background task manager like Celery or FastAPI's BackgroundTasks
    import asyncio
    asyncio.create_task(semantic_cache_manager.set_cached_response(prompt, llm_response_content))
    # await semantic_cache_manager.set_cached_response(prompt, llm_response_content) # If you need to ensure it's cached

    return ChatResponse(prompt=prompt, response=llm_response_content, source="llm")

@app.get("/health")
async def health_check():
    # Basic health check, can be expanded to check Redis connection too
    if semantic_cache_manager.redis_client and semantic_cache_manager.redis_client.ping():
        redis_status = "connected"
    else:
        redis_status = "disconnected"
    return {"status": "healthy", "redis_status": redis_status, "app_name": settings.APP_NAME}

# To run: uvicorn app.main:app --reload --port 8000
```

### 6.2. Request/Response Models (Pydantic) (`app/models/pydantic_models.py`)
```python
# app/models/pydantic_models.py
from pydantic import BaseModel
from typing import Literal

class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    prompt: str
    response: str
    source: Literal["cache", "llm"]
```

### 6.3. API Endpoint: `/chat_semantic_cache`
Defined in `app/main.py`.

---

## 7. Phase 4: Dockerization

### 7.1. `Dockerfile` for the Application
```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Set environment variables (can be overridden in docker-compose)
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies if any (e.g., for certain ML models)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY ./app /app/app
COPY ./.env /app/.env # Copy .env file if it contains non-sensitive defaults or for local dev

# Expose port
EXPOSE 8000

# Command to run the application
# Note: For production, you might use gunicorn instead of uvicorn directly for more workers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 7.2. `docker-compose.yml`
```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: semantic_cache_app
    ports:
      - "8000:8000"
    volumes:
      - ./app:/app/app # Mount app directory for live reloading during development
      - ./.env:/app/.env # Mount .env file
    environment:
      # Environment variables can be set here or read from the .env file copied in Dockerfile
      # These will override .env if both are present and sourced by the app
      - REDIS_HOST=redis_cache_service # Critical: Service name for Redis
      # - HF_TOKEN=${HF_TOKEN} # Pass through from host .env if needed
      # - LLM_ENDPOINT_URL=${LLM_ENDPOINT_URL}
      # - HF_EMBEDDING_ENDPOINT_URL=${HF_EMBEDDING_ENDPOINT_URL}
    depends_on:
      - redis_cache_service
    restart: unless-stopped

  redis_cache_service:
    image: redis/redis-stack-server:latest # Includes RediSearch
    container_name: redis_semantic_cache
    ports:
      - "6379:6379" # For local client connection to Redis
      # - "8001:8001" # RedisInsight GUI (optional)
    volumes:
      - redis_data:/data # Persistent storage for Redis data
    restart: unless-stopped

volumes:
  redis_data: # Defines the named volume for Redis persistence
```
**Note:** For `HF_TOKEN` and other secrets in `docker-compose.yml`, it's better to have them in a host `.env` file and let Docker Compose pick them up automatically if the `environment` section references them as `${VARIABLE_NAME}` without a default. Docker Compose will look for a `.env` file in the same directory as the `docker-compose.yml`.

---

## 8. Phase 5: Testing Strategy

### 8.1. Unit Tests
*   Test embedding generation (`EmbeddingService`).
*   Test Redis index creation logic (mock Redis client).
*   Test cache get/set logic in `CacheManager` (mock Redis, mock embedding service).
*   Test Pydantic models.

### 8.2. Integration Tests
*   Test API endpoint (`/chat_semantic_cache`) with a live (Dockerized) Redis instance.
    *   Scenario 1: Cache miss -> LLM call -> Cache set.
    *   Scenario 2: Cache hit (exact prompt).
    *   Scenario 3: Cache hit (semantically similar prompt).
    *   Scenario 4: Cache miss (semantically different prompt).
*   Test health check endpoint.

### 8.3. Performance Tests (Cache Hit/Miss)
*   Measure response times for cache hits vs. cache misses.
*   Simulate multiple requests to observe caching behavior.

---

## 9. Phase 6: Deployment (Local) & Usage

### 9.1. Running with Docker Compose
```bash
# Ensure Docker Desktop is running
# In the project root directory (where docker-compose.yml is):
docker-compose up --build
```
To run in detached mode: `docker-compose up --build -d`
To stop: `docker-compose down`

### 9.2. Example API Calls
Using `curl` or a tool like Postman/Insomnia:
```bash
# Example: Cache Miss then Hit
curl -X POST "http://localhost:8000/chat_semantic_cache" \
-H "Content-Type: application/json" \
-d '{
  "prompt": "What is the capital of France?"
}'

# Expected output (first time, source: llm):
# {"prompt":"What is the capital of France?","response":"...Paris...","source":"llm"}

curl -X POST "http://localhost:8000/chat_semantic_cache" \
-H "Content-Type: application/json" \
-d '{
  "prompt": "Tell me the capital city of France."
}'
# Expected output (second time, source: cache, if semantically similar enough):
# {"prompt":"Tell me the capital city of France.","response":"...Paris...","source":"cache"}
```

---

## 10. Phase 7: Documentation

### 10.1. README.md
*   Project overview and goals.
*   Technology stack.
*   Setup instructions (virtual env, dependencies).
*   Configuration (how to set up `.env`).
*   How to run (locally with `uvicorn`, with `docker-compose`).
*   API endpoint documentation (request/response examples).
*   Testing instructions.
*   Directory structure explanation.

### 10.2. Code Comments
*   Ensure all functions and complex logic blocks are well-commented.
*   Explain choices made in the code.

---

## 11. Future Enhancements & Considerations
*   **Cache Invalidation:** Implement more sophisticated cache invalidation (e.g., if underlying knowledge for LLM changes). Currently relies on TTL.
*   **Scalability:**
    *   Use a more robust message queue (e.g., Celery with RabbitMQ/Redis) for background tasks like `set_cached_response`.
    *   Consider HNSW index for RediSearch for larger datasets for faster vector search.
*   **Advanced Similarity:** Explore different vector distance metrics or re-ranking if needed.
*   **Hybrid Search:** Combine keyword search with vector search in Redis for even better retrieval for some use cases.
*   **Batching:** Batch embeddings and Redis writes for higher throughput if handling many concurrent requests.
*   **Monitoring & Analytics:** Integrate logging for cache hit/miss rates, latencies. Use Redis monitoring tools.
*   **Error Handling:** More robust error handling for external service calls (LLM, Redis).
*   **Security:** Add API key authentication for the FastAPI endpoints.
*   **Multiple Namespaces/Indexes:** If caching for different LLMs or different types of prompts, allow for multiple Redis indexes or namespaces within the cache manager.
*   **Async All The Way:** Ensure all I/O bound operations (Redis calls, LLM calls) are truly asynchronous using `async/await` correctly throughout the FastAPI request lifecycle. (Current LLM service is a placeholder, ensure real one is async).
``` 