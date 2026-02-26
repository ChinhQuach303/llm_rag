import uuid
from sqlalchemy import Column, String, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from core.database import Base

class DocumentRecord(Base):
    """
    SQLAlchemy Model mapping to the 'documents' table in the relational database.
    Stores the raw, hierarchical DeepSeek JSON before it gets chunked/embedded.
    """
    __tablename__ = 'documents'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(String, nullable=False, index=True)
    filename = Column(String, nullable=False)
    status = Column(String, nullable=False, default="PROCESSING") # PROCESSING, COMPLETED, FAILED
    raw_content = Column(JSONB, nullable=True) # The structured Markdown JSON
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
