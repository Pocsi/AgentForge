import logging
from typing import Dict, Any, List, Optional
import networkx as nx
import json
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class NetworkAnalyzer:
    """Network topology and performance analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.topology_type = config.get("topology_type", "mesh")
        self.connection_timeout = config.get("connection_timeout", 5)
        self.retry_attempts = config.get("retry_attempts", 3)
        self.optimization_interval = config.get("optimization_interval", 300)
        
        # Initialize network graph
        self.graph = nx.Graph()
        
        # Initialize performance metrics
        self.metrics = {}
    
    async def analyze_topology(self) -> Dict[str, Any]:
        """Analyze network topology"""
        try:
            # Get current topology
            topology = self._get_current_topology()
            
            # Analyze connectivity
            connectivity = self._analyze_connectivity()
            
            # Analyze performance
            performance = self._analyze_performance()
            
            return {
                "topology": topology,
                "connectivity": connectivity,
                "performance": performance,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing topology: {str(e)}")
            return {}
    
    async def create_connections(
        self,
        node: Dict[str, Any],
        topology: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create optimal connections for a node"""
        try:
            node_id = node.get("id")
            node_type = node.get("type")
            
            # Add node to graph if not exists
            if not self.graph.has_node(node_id):
                self.graph.add_node(
                    node_id,
                    type=node_type,
                    performance=node.get("performance", 0)
                )
            
            # Find optimal connections
            connections = []
            for other_node in self.graph.nodes():
                if other_node != node_id:
                    # Calculate connection score
                    score = self._calculate_connection_score(
                        node_id,
                        other_node,
                        topology
                    )
                    
                    if score > 0.7:  # Only create high-quality connections
                        connection = {
                            "source": node_id,
                            "target": other_node,
                            "score": score,
                            "type": "optimal",
                            "status": "active"
                        }
                        connections.append(connection)
                        
                        # Add edge to graph
                        self.graph.add_edge(
                            node_id,
                            other_node,
                            score=score,
                            created_at=datetime.now().isoformat()
                        )
            
            return connections
            
        except Exception as e:
            logger.error(f"Error creating connections: {str(e)}")
            return []
    
    async def analyze_performance(
        self,
        node_metrics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze network performance"""
        try:
            if not node_metrics:
                return {}
            
            # Calculate aggregate metrics
            performance_metrics = {
                "average_latency": 0,
                "bandwidth_utilization": 0,
                "error_rate": 0,
                "node_count": len(node_metrics)
            }
            
            # Calculate metrics for each node
            for metrics in node_metrics:
                current_metrics = metrics.get("current_metrics", {})
                
                # Update aggregate metrics
                performance_metrics["average_latency"] += current_metrics.get("latency", 0)
                performance_metrics["bandwidth_utilization"] += current_metrics.get("bandwidth_utilization", 0)
                performance_metrics["error_rate"] += current_metrics.get("error_rate", 0)
            
            # Calculate averages
            node_count = performance_metrics["node_count"]
            if node_count > 0:
                performance_metrics["average_latency"] /= node_count
                performance_metrics["bandwidth_utilization"] /= node_count
                performance_metrics["error_rate"] /= node_count
            
            # Calculate network health score
            performance_metrics["health_score"] = self._calculate_health_score(
                performance_metrics
            )
            
            return performance_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing performance: {str(e)}")
            return {}
    
    async def update_connections(
        self,
        connections: List[Dict[str, Any]],
        optimizations: List[Dict[str, Any]]
    ) -> None:
        """Update network connections based on optimizations"""
        try:
            # Update edge weights based on optimizations
            for optimization in optimizations:
                protocol = optimization.get("protocol")
                improvement = optimization.get("improvement", 0)
                
                # Update affected edges
                for edge in self.graph.edges():
                    if self._is_edge_affected(edge, protocol):
                        current_weight = self.graph[edge[0]][edge[1]].get("score", 0)
                        new_weight = min(1.0, current_weight * (1 + improvement))
                        self.graph[edge[0]][edge[1]]["score"] = new_weight
            
            # Remove low-quality connections
            edges_to_remove = [
                edge for edge in self.graph.edges()
                if self.graph[edge[0]][edge[1]].get("score", 0) < 0.5
            ]
            self.graph.remove_edges_from(edges_to_remove)
            
        except Exception as e:
            logger.error(f"Error updating connections: {str(e)}")
    
    def _get_current_topology(self) -> Dict[str, Any]:
        """Get current network topology"""
        try:
            return {
                "nodes": list(self.graph.nodes(data=True)),
                "edges": list(self.graph.edges(data=True)),
                "density": nx.density(self.graph),
                "average_degree": sum(dict(self.graph.degree()).values()) / len(self.graph) if len(self.graph) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting current topology: {str(e)}")
            return {}
    
    def _analyze_connectivity(self) -> Dict[str, Any]:
        """Analyze network connectivity"""
        try:
            if not self.graph:
                return {}
            
            return {
                "is_connected": nx.is_connected(self.graph),
                "number_of_components": nx.number_connected_components(self.graph),
                "average_shortest_path": nx.average_shortest_path_length(self.graph) if nx.is_connected(self.graph) else float('inf'),
                "diameter": nx.diameter(self.graph) if nx.is_connected(self.graph) else float('inf')
            }
            
        except Exception as e:
            logger.error(f"Error analyzing connectivity: {str(e)}")
            return {}
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze network performance metrics"""
        try:
            if not self.graph:
                return {}
            
            # Calculate edge performance metrics
            edge_metrics = []
            for edge in self.graph.edges(data=True):
                edge_metrics.append(edge[2].get("score", 0))
            
            return {
                "average_edge_score": np.mean(edge_metrics) if edge_metrics else 0,
                "min_edge_score": min(edge_metrics) if edge_metrics else 0,
                "max_edge_score": max(edge_metrics) if edge_metrics else 0,
                "edge_score_std": np.std(edge_metrics) if edge_metrics else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing performance: {str(e)}")
            return {}
    
    def _calculate_connection_score(
        self,
        source: str,
        target: str,
        topology: Dict[str, Any]
    ) -> float:
        """Calculate connection score between nodes"""
        try:
            # Get node data
            source_data = self.graph.nodes[source]
            target_data = self.graph.nodes[target]
            
            # Calculate base score
            base_score = min(
                source_data.get("performance", 0),
                target_data.get("performance", 0)
            )
            
            # Adjust score based on topology
            if topology.get("density", 0) > 0.8:
                # High density network - prefer shorter paths
                path_length = nx.shortest_path_length(self.graph, source, target)
                base_score *= (1 - path_length / len(self.graph))
            
            return base_score
            
        except Exception as e:
            logger.error(f"Error calculating connection score: {str(e)}")
            return 0.0
    
    def _is_edge_affected(
        self,
        edge: tuple,
        protocol: str
    ) -> bool:
        """Check if an edge is affected by a protocol optimization"""
        try:
            # In a real implementation, this would check protocol-specific criteria
            # For now, we'll use a simple check
            return True
            
        except Exception as e:
            logger.error(f"Error checking edge effect: {str(e)}")
            return False
    
    def _calculate_health_score(
        self,
        metrics: Dict[str, Any]
    ) -> float:
        """Calculate network health score"""
        try:
            # Normalize metrics
            latency_score = 1 - min(1.0, metrics.get("average_latency", 0) / 1000)
            bandwidth_score = 1 - metrics.get("bandwidth_utilization", 0)
            error_score = 1 - metrics.get("error_rate", 0)
            
            # Calculate weighted average
            weights = {
                "latency": 0.4,
                "bandwidth": 0.3,
                "error": 0.3
            }
            
            health_score = (
                weights["latency"] * latency_score +
                weights["bandwidth"] * bandwidth_score +
                weights["error"] * error_score
            )
            
            return health_score
            
        except Exception as e:
            logger.error(f"Error calculating health score: {str(e)}")
            return 0.0 