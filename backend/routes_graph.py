from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
# We assume graph_engine has been instantiated somewhere or we instantiate here.
from .graph_engine import GraphEngine

router = APIRouter(prefix="/api/graph", tags=["Graph"])

graph_engine = GraphEngine()

@router.get("/status")
async def get_graph_status():
    """Check Neo4j DB connectivity"""
    return {"connected": graph_engine.test_connection()}

@router.get("/mule-rings")
async def get_mule_rings():
    """Detect and return suspicious mule rings"""
    try:
        rings = graph_engine.detect_mule_rings()
        return {"mule_rings": rings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/network/{user_id}")
async def get_user_network(user_id: str):
    """Get the network neighborhood for a specific user"""
    try:
        network = graph_engine.get_user_network(user_id)
        return {"user_id": user_id, "network": network}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
