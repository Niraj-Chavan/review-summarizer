import networkx as nx
import community as community_louvain # python-louvain
import pandas as pd
from typing import List, Dict, Any

class CollusionGraph:
    def __init__(self):
        self.graph = nx.Graph()

    def build_graph(self, reviews: List[Dict[str, Any]]):
        """
        Builds a bipartite graph of reviewers and products.
        An edge exists if a reviewer reviewed a product.
        Edge weights can represent burstiness or shared timestamps.
        """
        for rev in reviews:
            r_id = f"user_{rev['reviewer_id']}"
            p_id = f"prod_{rev['product_id']}"
            
            self.graph.add_node(r_id, bipartite=0)
            self.graph.add_node(p_id, bipartite=1)
            
            # Simple unweighted edge for now, can add timestamp proximity logic
            self.graph.add_edge(r_id, p_id)

    def detect_collusion_clusters(self) -> Dict[str, float]:
        """
        Runs Louvain community detection.
        Returns a dictionary mapping reviewer_id to a collusion risk score (0 to 1).
        Risk is higher if the community has an abnormally high density 
        (e.g. many users reviewing the exact same small set of products).
        """
        if len(self.graph.nodes) == 0:
            return {}

        # Louvain works on unipartite projections or the bipartite graph directly.
        # Running on the full graph finds communities of users + products.
        partition = community_louvain.best_partition(self.graph)
        
        # Analyze communities
        community_stats = {}
        for node, comm_id in partition.items():
            if comm_id not in community_stats:
                community_stats[comm_id] = {'users': 0, 'products': 0, 'edges': 0}
            
            if str(node).startswith("user_"):
                community_stats[comm_id]['users'] += 1
            else:
                community_stats[comm_id]['products'] += 1

        # Calculate bipartite density for each community
        # Density = edges / (users * products). High density = high collusion risk
        community_risk = {}
        for comm_id, stats in community_stats.items():
            u = stats['users']
            p = stats['products']
            if u > 1 and p > 0:
                # Count intra-community edges
                subgraph = self.graph.subgraph([n for n, c in partition.items() if c == comm_id])
                density = subgraph.number_of_edges() / (u * p)
                community_risk[comm_id] = min(density, 1.0)
            else:
                community_risk[comm_id] = 0.0

        # Map risk back to reviewers
        reviewer_risk = {}
        for node, comm_id in partition.items():
            if str(node).startswith("user_"):
                r_id = str(node).replace("user_", "")
                reviewer_risk[r_id] = community_risk[comm_id]

        return reviewer_risk
