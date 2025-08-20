"""
Account Configuration Manager

Handles Alpaca account configurations and settings
Reference: https://docs.alpaca.markets/reference/patchaccountconfig-1
"""

import requests
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

class AccountConfigurator:
    """Manages Alpaca account configurations"""
    
    def __init__(self, account_manager):
        self.account_manager = account_manager
        self.logger = logging.getLogger(__name__)
    
    def get_account_configurations(self) -> Dict[str, Any]:
        """Get current account configurations"""
        try:
            response = requests.get(
                f"{self.account_manager.base_url}/v2/account/configurations",
                headers=self.account_manager.headers,
                timeout=10
            )
            response.raise_for_status()
            
            configs = response.json()
            self.logger.info("Account configurations retrieved successfully")
            
            return configs
            
        except Exception as e:
            self.logger.error(f"Error retrieving account configurations: {e}")
            return {"error": str(e)}
    
    def update_account_configurations(self, configurations: Dict[str, Any]) -> Dict[str, Any]:
        """Update account configurations
        
        Available configurations:
        - day_trade_margin_call: EQUITY or CASH
        - trade_confirm_email: EMAIL or ALL or NONE
        - suspend_trade: true or false
        - no_shorting: true or false
        - fractional_trading: true or false
        - max_margin_multiplier: float (1.0 to 4.0)
        - pdt_check: ENTRY or EXIT or BOTH or NONE
        - trading_hours: STANDARD or EXTENDED
        """
        
        # Validate configuration values
        valid_configs = self._validate_configurations(configurations)
        if 'error' in valid_configs:
            return valid_configs
        
        try:
            response = requests.patch(
                f"{self.account_manager.base_url}/v2/account/configurations",
                headers=self.account_manager.headers,
                json=configurations,
                timeout=10
            )
            response.raise_for_status()
            
            updated_configs = response.json()
            self.logger.info("Account configurations updated successfully")
            
            return updated_configs
            
        except Exception as e:
            self.logger.error(f"Error updating account configurations: {e}")
            return {"error": str(e)}
    
    def _validate_configurations(self, configurations: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration parameters"""
        valid_options = {
            'day_trade_margin_call': ['EQUITY', 'CASH'],
            'trade_confirm_email': ['EMAIL', 'ALL', 'NONE'],
            'suspend_trade': [True, False],
            'no_shorting': [True, False],
            'fractional_trading': [True, False],
            'pdt_check': ['ENTRY', 'EXIT', 'BOTH', 'NONE'],
            'trading_hours': ['STANDARD', 'EXTENDED']
        }
        
        errors = []
        
        for key, value in configurations.items():
            if key in valid_options:
                if key == 'max_margin_multiplier':
                    # Special validation for margin multiplier
                    try:
                        float_val = float(value)
                        if not (1.0 <= float_val <= 4.0):
                            errors.append(f"max_margin_multiplier must be between 1.0 and 4.0")
                    except (ValueError, TypeError):
                        errors.append(f"max_margin_multiplier must be a number")
                elif value not in valid_options[key]:
                    errors.append(f"Invalid value for {key}: {value}. Valid options: {valid_options[key]}")
            elif key != 'max_margin_multiplier':
                errors.append(f"Unknown configuration parameter: {key}")
        
        if errors:
            return {"error": f"Validation errors: {'; '.join(errors)}"}
        
        return {"valid": True}
    
    def enable_extended_hours_trading(self) -> Dict[str, Any]:
        """Enable extended hours trading"""
        return self.update_account_configurations({
            'trading_hours': 'EXTENDED'
        })
    
    def disable_extended_hours_trading(self) -> Dict[str, Any]:
        """Disable extended hours trading (standard hours only)"""
        return self.update_account_configurations({
            'trading_hours': 'STANDARD'
        })
    
    def enable_fractional_trading(self) -> Dict[str, Any]:
        """Enable fractional share trading"""
        return self.update_account_configurations({
            'fractional_trading': True
        })
    
    def disable_fractional_trading(self) -> Dict[str, Any]:
        """Disable fractional share trading"""
        return self.update_account_configurations({
            'fractional_trading': False
        })
    
    def set_pdt_check(self, check_type: str = 'BOTH') -> Dict[str, Any]:
        """Set Pattern Day Trader check
        
        Options:
        - ENTRY: Check before order entry
        - EXIT: Check before order exit
        - BOTH: Check on both entry and exit
        - NONE: No PDT checking
        """
        return self.update_account_configurations({
            'pdt_check': check_type.upper()
        })
    
    def set_margin_multiplier(self, multiplier: float) -> Dict[str, Any]:
        """Set maximum margin multiplier (1.0 to 4.0)"""
        return self.update_account_configurations({
            'max_margin_multiplier': multiplier
        })
    
    def enable_shorting(self) -> Dict[str, Any]:
        """Enable short selling"""
        return self.update_account_configurations({
            'no_shorting': False
        })
    
    def disable_shorting(self) -> Dict[str, Any]:
        """Disable short selling"""
        return self.update_account_configurations({
            'no_shorting': True
        })
    
    def set_trade_confirmation_email(self, email_type: str = 'ALL') -> Dict[str, Any]:
        """Set trade confirmation email preferences
        
        Options:
        - EMAIL: Send to email address
        - ALL: Send all confirmations
        - NONE: No email confirmations
        """
        return self.update_account_configurations({
            'trade_confirm_email': email_type.upper()
        })
    
    def suspend_trading(self) -> Dict[str, Any]:
        """Suspend trading on the account"""
        return self.update_account_configurations({
            'suspend_trade': True
        })
    
    def resume_trading(self) -> Dict[str, Any]:
        """Resume trading on the account"""
        return self.update_account_configurations({
            'suspend_trade': False
        })
    
    def get_configuration_summary(self) -> str:
        """Get formatted configuration summary"""
        configs = self.get_account_configurations()
        
        if 'error' in configs:
            return f"Error retrieving configurations: {configs['error']}"
        
        summary = f"""
🌍 GAUSS WORLD TRADER - ACCOUNT CONFIGURATIONS
=============================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

TRADING SETTINGS:
----------------
• Trading Hours: {configs.get('trading_hours', 'N/A')}
• Extended Hours: {'Enabled' if configs.get('trading_hours') == 'EXTENDED' else 'Disabled'}
• Fractional Trading: {'Enabled' if configs.get('fractional_trading', False) else 'Disabled'}
• Short Selling: {'Disabled' if configs.get('no_shorting', False) else 'Enabled'}
• Trading Suspended: {'Yes' if configs.get('suspend_trade', False) else 'No'}

MARGIN SETTINGS:
---------------
• Day Trade Margin Call: {configs.get('day_trade_margin_call', 'N/A')}
• Max Margin Multiplier: {configs.get('max_margin_multiplier', 'N/A')}
• PDT Check: {configs.get('pdt_check', 'N/A')}

NOTIFICATIONS:
-------------
• Trade Confirmation Email: {configs.get('trade_confirm_email', 'N/A')}

Using: {"Paper Trading" if "paper" in self.account_manager.base_url else "Live Trading"}
"""
        
        return summary
    
    def apply_conservative_settings(self) -> Dict[str, Any]:
        """Apply conservative trading settings"""
        conservative_config = {
            'trading_hours': 'STANDARD',
            'fractional_trading': False,
            'no_shorting': True,
            'pdt_check': 'BOTH',
            'max_margin_multiplier': 1.0,
            'trade_confirm_email': 'ALL'
        }
        
        result = self.update_account_configurations(conservative_config)
        
        if 'error' not in result:
            self.logger.info("Conservative trading settings applied")
        
        return result
    
    def apply_aggressive_settings(self) -> Dict[str, Any]:
        """Apply aggressive trading settings"""
        aggressive_config = {
            'trading_hours': 'EXTENDED',
            'fractional_trading': True,
            'no_shorting': False,
            'pdt_check': 'NONE',
            'max_margin_multiplier': 4.0,
            'trade_confirm_email': 'NONE'
        }
        
        result = self.update_account_configurations(aggressive_config)
        
        if 'error' not in result:
            self.logger.info("Aggressive trading settings applied")
        
        return result