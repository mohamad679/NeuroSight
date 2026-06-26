import asyncio
import importlib
from unittest.mock import AsyncMock, patch
from httpx import ASGITransport, AsyncClient

async def main():
    mock_redis = AsyncMock()
    mock_audio = AsyncMock()
    mock_llm = AsyncMock()
    mock_neo4j = AsyncMock()
    mock_vision = AsyncMock()
    mock_migration = AsyncMock(return_value={"status": "current"})
    
    with (
        patch("app.core.redis.redis_service.connect", mock_redis.connect) as p1,
        patch("app.core.redis.redis_service.close", mock_redis.close),
        patch("app.core.database.check_migration_status", mock_migration),
        patch("app.services.audio_processing.audio_processing_service.close", mock_audio.close),
        patch("app.services.llm.llm_service.close", mock_llm.close),
        patch("app.services.neo4j_graph.neo4j_graph_service.close", mock_neo4j.close),
        patch("app.services.vision.vision_service.close", mock_vision.close),
        patch("app.worker.background_jobs_health", return_value={"status": "healthy"}),
    ):
        main_module = importlib.import_module("app.main")
        app = main_module.app
        
        print("redis_service inside app.main:", main_module.redis_service)
        print("redis_service.connect is mock:", main_module.redis_service.connect == mock_redis.connect)
        
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            print("Entering context manager")
            response = await ac.get("/")
            print("Response status:", response.status_code)
            
        print("mock_redis.connect call count:", mock_redis.connect.call_count)
        print("mock_redis.close call count:", mock_redis.close.call_count)

if __name__ == "__main__":
    asyncio.run(main())
