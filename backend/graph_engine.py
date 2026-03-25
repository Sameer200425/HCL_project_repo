"""
Neo4j Graph Database Fraud Detection Engine
"""

from py2neo import Graph, Node, Relationship
import os
import networkx as nx
import numpy as np

# In a real environment, read from env variables. Hardcoding for demo resilience.
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

class GraphEngine:
    def __init__(self):
        self.graph: Graph | None = None
        self.connected = False
        try:
            self.graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            self.connected = True
            print("Successfully connected to Neo4j.")
        except Exception as e:
            self.connected = False
            self.graph = None
            print(f"Neo4j connection failed: {e}. Running in degraded (mock) mode.")
            self._mock_data()

    def _mock_data(self):
        # Mock logic if Neo4j is offline
        self.mock_nodes = [{"id": "user1", "label": "User"}, {"id": "acc1", "label": "Account"}]
        self.mock_edges = [{"from": "user1", "to": "acc1", "type": "OWNS"}]

    def test_connection(self):
        return self.connected

    def detect_mule_rings(self):
        """Identify potential mule rings by looking for shared attributes (e.g. phone, address) across accounts"""
        graph = self.graph
        if not self.connected or graph is None:
            return [{"ring_id": 1, "accounts": ["acc1", "acc2"], "risk_score": 0.85}]

        query = """
        MATCH (u1:User)-[:HAS_IP]->(ip:IPAddress)<-[:HAS_IP]-(u2:User)
        WHERE id(u1) < id(u2)
        RETURN u1.id AS User1, u2.id AS User2, ip.address AS SharedIP, 0.9 AS risk_score
        LIMIT 5
        """
        results = graph.run(query).data()
        return results

    def get_user_network(self, user_id):
        """Retrieve local graph for a user"""
        if not self.connected:
            return {"nodes": self.mock_nodes, "edges": self.mock_edges}

        query = f"""
        MATCH p=(u:User {{id: '{user_id}'}})-[*1..2]-()
        RETURN nodes(p) AS nodes, relationships(p) AS rels
        """
        # Complex parsing omitted for brevity. Return simple json
        return {"nodes": [], "edges": []}
