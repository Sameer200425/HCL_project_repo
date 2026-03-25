import os
from py2neo import Graph, Node, Relationship

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

def seed_database():
    try:
        print("Connecting to Neo4j...")
        graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        print("Clearing existing data...")
        graph.delete_all()
        
        print("Creating Nodes and Relationships...")
        u1 = Node("User", id="u100", name="Alice")
        u2 = Node("User", id="u101", name="Bob")
        u3 = Node("User", id="u102", name="Charlie")
        
        ip1 = Node("IPAddress", address="192.168.1.100")
        ip2 = Node("IPAddress", address="10.0.0.5")
        
        rel1 = Relationship(u1, "HAS_IP", ip1)
        rel2 = Relationship(u2, "HAS_IP", ip1)  # Shared IP -> Mule Ring risk
        rel3 = Relationship(u3, "HAS_IP", ip2)
        
        tx = graph.begin()
        tx.create(u1 | u2 | u3 | ip1 | ip2 | rel1 | rel2 | rel3)
        graph.commit(tx)
        
        print("Data seeded successfully!")
    except Exception as e:
        print(f"Failed to connect to Neo4j. Ensure it is running. Error: {e}")

if __name__ == "__main__":
    seed_database()
