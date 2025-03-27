import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from strategic_planning import StrategicPlanner
from config import get_monetization_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Run strategic analysis to find optimal path"""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get configuration
        config = get_monetization_config()
        
        # Initialize strategic planner
        planner = StrategicPlanner(config)
        
        logger.info("Starting strategic analysis...")
        
        # Analyze opportunities
        analysis = await planner.analyze_opportunities()
        
        # Save results
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"strategic_analysis_{timestamp}.json"
        
        with open(output_file, "w") as f:
            json.dump(analysis, f, indent=2)
        
        # Print summary
        print("\n=== Strategic Analysis Summary ===")
        print(f"Analysis saved to: {output_file}")
        
        recommendations = analysis.get("recommendations", [])
        timeline = analysis.get("timeline", {})
        
        print("\nTop Recommendations:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"\n{i}. {rec['type'].replace('_', ' ').title()}")
            print(f"   Earning Potential: ${rec.get('earning_potential', 0):.2f}")
            print(f"   Time to First Earnings: {rec.get('time_to_first_earnings', 0)} days")
            print(f"   Risk Level: {rec.get('risk_level', 'medium')}")
        
        print("\nTimeline:")
        print(f"Total Duration: {timeline.get('total_duration', 0)} days")
        print(f"Expected Earnings: ${timeline.get('expected_earnings', 0):.2f}")
        
        print("\nNext Steps:")
        if recommendations:
            best_option = recommendations[0]
            print(f"1. Start with {best_option['type'].replace('_', ' ').title()}")
            print(f"2. Allocate ${best_option.get('learning_cost', 0) or best_option.get('startup_cost', 0):.2f} for initial setup")
            print(f"3. Focus on getting first earnings within {best_option.get('time_to_first_earnings', 0)} days")
        
    except Exception as e:
        logger.error(f"Error in strategic analysis: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 