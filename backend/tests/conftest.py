import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
import asyncio
from main import app
from unittest.mock import patch

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
