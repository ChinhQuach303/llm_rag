import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO
from main import app
from core.security import get_current_tenant

@pytest.mark.asyncio
async def test_ingest_endpoint(client):
    # Mock the celery task and Minio storage to avoid real external calls during integration test
    with patch("api.routes.ingest.process_document_task.delay") as mock_delay, \
         patch("api.routes.ingest.MinioStorageService") as mock_minio:
        
        # Mock task return
        mock_task = MagicMock()
        mock_task.id = "mock-task-id-123"
        mock_delay.return_value = mock_task

        # Create dummy file content
        file_content = b"dummy pdf content"
        files = {"file": ("test.pdf", BytesIO(file_content), "application/pdf")}
        data = {"tenant_id": "tenant_test"}

        app.dependency_overrides[get_current_tenant] = lambda: "tenant_test"

        response = await client.post("/v1/ingest", files=files, data=data)

        assert response.status_code == 200
        result = response.json()
        assert result["task_id"] == "mock-task-id-123"
        assert result["status"] == "QUEUED_TO_WORKER"
        assert "test.pdf" in result["message"]

        # Verify mocks were called
        mock_minio.return_value.upload_file.assert_called_once()
        mock_delay.assert_called_once()


@pytest.mark.asyncio
async def test_query_endpoint(client):
    # Mock the RAG pipeline service
    with patch("api.dependencies._pipeline_instance") as mock_pipeline_instance:
        mock_pipeline = MagicMock()
        
        # Generator for streaming response
        async def mock_stream_response(*args, **kwargs):
            yield "This is a mocked "
            yield "streaming response."

        mock_pipeline.answer_query.return_value = mock_stream_response()
        
        # We need to test the dependency injection by overriding it
        from main import app
        from api.dependencies import get_rag_pipeline
        
        app.dependency_overrides[get_rag_pipeline] = lambda: mock_pipeline
        app.dependency_overrides[get_current_tenant] = lambda: "tenant_test"

        data = {
            "query": "What is the summary?",
            "tenant_id": "tenant_test",
            "history": []
        }

        response = await client.post("/v1/query", json=data)

        assert response.status_code == 200
        # The response is a stream, so we can check the text
        assert response.text == "This is a mocked streaming response."
        
        # Clean up overrides
        app.dependency_overrides.clear()
