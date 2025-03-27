import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from network_swarm import NetworkSwarm
from config import get_monetization_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Run the enhanced network swarm strategy"""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get configuration
        config = get_monetization_config()
        
        # Initialize network swarm
        swarm = NetworkSwarm(config)
        
        logger.info("Starting network swarm strategy...")
        
        # Execute swarm strategy
        result = await swarm.execute_swarm_strategy()
        
        # Save results
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"swarm_results_{timestamp}.json"
        
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        
        # Print summary
        print("\n=== Network Swarm Strategy Results ===")
        print(f"Results saved to: {output_file}")
        
        # Print initialization results
        init_result = result.get("initialization", {})
        print("\nNetwork Initialization:")
        print(f"Edge Nodes: {len(init_result.get('edge_nodes', []))}")
        print(f"Network Connections: {len(init_result.get('connections', []))}")
        print(f"AI Protocols: {len(init_result.get('protocols', []))}")
        
        # Print optimization results
        opt_result = result.get("optimization", {})
        print("\nNetwork Optimization:")
        print(f"Optimized Protocols: {len(opt_result.get('optimizations', []))}")
        print(f"Performance Metrics: {json.dumps(opt_result.get('metrics', {}), indent=2)}")
        
        # Print execution results
        exec_result = result.get("execution", {})
        print("\nStrategy Execution:")
        print(f"Opportunities Found: {len(exec_result.get('opportunities', []))}")
        print(f"Trades Executed: {len(exec_result.get('trades', []))}")
        print(f"Performance: {json.dumps(exec_result.get('performance', {}), indent=2)}")
        
        # Print next steps
        print("\nNext Steps:")
        if exec_result.get("trades"):
            print("1. Monitor trade performance")
            print("2. Adjust network parameters based on results")
            print("3. Scale successful strategies")
        else:
            print("1. Review network configuration")
            print("2. Adjust AI protocols")
            print("3. Explore alternative opportunities")
        
    except Exception as e:
        logger.error(f"Error in network swarm strategy: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 