"""
core/security.py
----------------
Manages inbound Authentication and Access Control checks.
Currently utilizes a stateless JWT Bearer token approach to extract tenant boundaries.
"""

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from core.config import JWT_SECRET_KEY, JWT_ALGORITHM
import logging

logger = logging.getLogger(__name__)

security = HTTPBearer()

def get_current_tenant(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Decodes the JWT token to extract the tenant_id.
    
    Why Stateless JWTs?
    Allows Horizontal Scaling of the FastAPI backend without needing a central DB
    to validate sessions on every single request. The `tenant_id` ensures strict 
    data-isolation inside Qdrant and Postgres logic.
    """
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        tenant_id = payload.get("tenant_id")
        if tenant_id is None:
            raise HTTPException(status_code=401, detail="Invalid token: tenant_id missing")
        return tenant_id
    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.PyJWTError as e:
        logger.warning(f"JWT Error: {e}")
        raise HTTPException(status_code=401, detail="Could not validate credentials")
