# Deployment & Infrastructure Instructions

## Configuration Management
- Use environment variables for secrets
- Pydantic Settings for configuration validation
- Separate configs for dev/staging/prod environments

## Service Architecture
- FastAPI for web services
- Include health check endpoints
- Implement graceful shutdown handling
- Use dependency injection for testability

## Monitoring & Observability
- Structured logging with correlation IDs
- Include metrics collection points
- Implement proper error tracking
- Use async logging to avoid blocking

## Docker Best Practices
- Multi-stage builds for smaller images
- Run as non-root user
- Use .dockerignore appropriately
- Pin base image versions

## Common Patterns
```python
# Service setup with proper lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    yield
    # Cleanup logic
```

## Security
- Input validation and sanitization
- Rate limiting for public endpoints
- HTTPS only for external communications
- Never log sensitive data