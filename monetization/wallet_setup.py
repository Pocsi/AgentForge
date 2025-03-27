import logging
from typing import Dict, Any, Optional
from web3 import Web3
from eth_account import Account
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class WalletSetup:
    """Handles wallet setup and configuration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.w3 = Web3(Web3.HTTPProvider('https://polygon-rpc.com'))
        self.wallet_dir = Path('wallets')
        self.wallet_dir.mkdir(exist_ok=True)
    
    def create_wallet(self, password: str) -> Dict[str, Any]:
        """Create a new wallet"""
        try:
            # Create account
            account = Account.create()
            
            # Encrypt private key
            encrypted_key = account.encrypt(password)
            
            # Save wallet
            wallet_path = self.wallet_dir / f"{account.address}.json"
            with open(wallet_path, 'w') as f:
                json.dump(encrypted_key, f)
            
            return {
                "address": account.address,
                "wallet_path": str(wallet_path),
                "status": "created"
            }
            
        except Exception as e:
            logger.error(f"Error creating wallet: {str(e)}")
            raise
    
    def load_wallet(self, address: str, password: str) -> Optional[Account]:
        """Load an existing wallet"""
        try:
            wallet_path = self.wallet_dir / f"{address}.json"
            if not wallet_path.exists():
                return None
            
            with open(wallet_path, 'r') as f:
                encrypted_key = json.load(f)
            
            # Decrypt private key
            private_key = Account.decrypt(encrypted_key, password)
            return Account.from_key(private_key)
            
        except Exception as e:
            logger.error(f"Error loading wallet: {str(e)}")
            return None
    
    def get_balance(self, address: str) -> float:
        """Get wallet balance in MATIC"""
        try:
            balance_wei = self.w3.eth.get_balance(address)
            return self.w3.from_wei(balance_wei, 'ether')
        except Exception as e:
            logger.error(f"Error getting balance: {str(e)}")
            return 0.0
    
    def check_gas_price(self) -> float:
        """Get current gas price in Gwei"""
        try:
            gas_price = self.w3.eth.gas_price
            return self.w3.from_wei(gas_price, 'gwei')
        except Exception as e:
            logger.error(f"Error getting gas price: {str(e)}")
            return 0.0
    
    def estimate_transaction_cost(self) -> float:
        """Estimate transaction cost in MATIC"""
        try:
            gas_price = self.w3.eth.gas_price
            gas_limit = self.config["goat"]["gas_limit"]
            cost_wei = gas_price * gas_limit
            return self.w3.from_wei(cost_wei, 'ether')
        except Exception as e:
            logger.error(f"Error estimating transaction cost: {str(e)}")
            return 0.0 