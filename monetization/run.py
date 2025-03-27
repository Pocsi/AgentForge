import asyncio
import logging
from typing import Dict, Any
import os
from dotenv import load_dotenv

from automation import MonetizationAutomation
from config import get_monetization_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Main function to run the monetization automation"""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get configuration
        config = get_monetization_config()
        
        # Initialize automation
        automation = MonetizationAutomation(config)
        
        # Start automation
        logger.info("Starting monetization automation...")
        await automation.start_automation()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 