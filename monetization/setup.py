import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv
import getpass

from config import get_monetization_config
from wallet_setup import WalletSetup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment() -> None:
    """Set up the environment for monetization automation"""
    try:
        # Load existing environment variables
        load_dotenv()
        
        # Get configuration
        config = get_monetization_config()
        
        # Initialize wallet setup
        wallet_setup = WalletSetup(config)
        
        # Check if wallet exists
        wallet_address = os.getenv("WALLET_ADDRESS")
        if not wallet_address:
            print("\n=== Wallet Setup ===")
            print("Creating new wallet...")
            password = getpass.getpass("Enter password for wallet encryption: ")
            wallet_info = wallet_setup.create_wallet(password)
            
            # Save wallet address to .env
            with open(".env", "a") as f:
                f.write(f"\nWALLET_ADDRESS={wallet_info['address']}")
            
            print(f"\nWallet created successfully!")
            print(f"Address: {wallet_info['address']}")
            print(f"Wallet file: {wallet_info['wallet_path']}")
            print("\nIMPORTANT: Save your wallet address and password securely!")
        
        # Check wallet balance
        balance = wallet_setup.get_balance(wallet_address)
        print(f"\nCurrent wallet balance: {balance:.4f} MATIC")
        
        # Check gas price
        gas_price = wallet_setup.check_gas_price()
        print(f"Current gas price: {gas_price:.2f} Gwei")
        
        # Estimate transaction cost
        tx_cost = wallet_setup.estimate_transaction_cost()
        print(f"Estimated transaction cost: {tx_cost:.4f} MATIC")
        
        # Check API keys
        goat_key = os.getenv("GOAT_API_KEY")
        mcp_key = os.getenv("MCP_API_KEY")
        
        if not goat_key or not mcp_key:
            print("\n=== API Keys Setup ===")
            if not goat_key:
                goat_key = input("Enter your GOAT API key: ")
                with open(".env", "a") as f:
                    f.write(f"\nGOAT_API_KEY={goat_key}")
            
            if not mcp_key:
                mcp_key = input("Enter your MCP API key: ")
                with open(".env", "a") as f:
                    f.write(f"\nMCP_API_KEY={mcp_key}")
        
        print("\n=== Setup Complete ===")
        print("Your environment is ready for monetization automation.")
        print("\nNext steps:")
        print("1. Fund your wallet with MATIC for gas fees")
        print("2. Fund your wallet with ETH for trading")
        print("3. Run 'python run.py' to start the automation")
        
    except Exception as e:
        logger.error(f"Error in setup: {str(e)}")
        raise

if __name__ == "__main__":
    setup_environment() 