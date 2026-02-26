import io
import logging
from minio import Minio
from minio.error import S3Error
from core.config import MINIO_URL, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET_NAME

logger = logging.getLogger(__name__)

class MinioStorageService:
    def __init__(self):
        try:
            self.client = Minio(
                MINIO_URL,
                access_key=MINIO_ACCESS_KEY,
                secret_key=MINIO_SECRET_KEY,
                secure=False # Local dev HTTP
            )
            self.bucket_name = MINIO_BUCKET_NAME
            
            # Ensure bucket exists
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info(f"Created MinIO bucket: {self.bucket_name}")
        except Exception as e:
            logger.error(f"MinIO Initialization failed: {e}")
            self.client = None

    def upload_file(self, object_name: str, file_data: bytes, content_type: str = "application/pdf") -> str:
        if not self.client:
            raise Exception("MinIO client not connected")
            
        try:
            self.client.put_object(
                self.bucket_name,
                object_name,
                io.BytesIO(file_data),
                len(file_data),
                content_type=content_type
            )
            return object_name
        except S3Error as e:
            logger.error(f"S3 Upload Error: {e}")
            raise

    def download_file(self, object_name: str, download_path: str):
        if not self.client:
            raise Exception("MinIO client not connected")
            
        try:
            self.client.fget_object(self.bucket_name, object_name, download_path)
            return download_path
        except S3Error as e:
            logger.error(f"S3 Download Error: {e}")
            raise
