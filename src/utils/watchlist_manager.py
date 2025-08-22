#!/usr/bin/env python3
"""
Watchlist Manager
Handles watchlist operations including reading, writing, adding, and removing symbols
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class WatchlistManager:
    """Manages watchlist operations with JSON persistence"""
    
    def __init__(self, watchlist_file: Optional[str] = None):
        """Initialize watchlist manager
        
        Args:
            watchlist_file: Path to watchlist JSON file. If None, uses default location.
        """
        if watchlist_file is None:
            # Default to project root
            project_root = Path(__file__).parent.parent.parent
            self.watchlist_file = project_root / "watchlist.json"
        else:
            self.watchlist_file = Path(watchlist_file)
        
        # Ensure the file exists
        self._ensure_watchlist_exists()
    
    def _ensure_watchlist_exists(self):
        """Ensure watchlist file exists with default content"""
        if not self.watchlist_file.exists():
            default_watchlist = {
                "watchlist": [
                    "AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", 
                    "AMZN", "META", "SPY", "QQQ", "VOO"
                ],
                "metadata": {
                    "created": datetime.now().strftime("%Y-%m-%d"),
                    "last_updated": datetime.now().strftime("%Y-%m-%d"),
                    "description": "Gauss World Trader Default Watchlist",
                    "version": "1.0"
                }
            }
            
            try:
                with open(self.watchlist_file, 'w') as f:
                    json.dump(default_watchlist, f, indent=2)
                logger.info(f"Created default watchlist at {self.watchlist_file}")
            except Exception as e:
                logger.error(f"Error creating default watchlist: {e}")
                raise
    
    def _load_watchlist(self) -> Dict:
        """Load watchlist from JSON file"""
        try:
            with open(self.watchlist_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Watchlist file not found: {self.watchlist_file}")
            self._ensure_watchlist_exists()
            return self._load_watchlist()
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing watchlist JSON: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading watchlist: {e}")
            raise
    
    def _save_watchlist(self, data: Dict):
        """Save watchlist to JSON file"""
        try:
            # Update metadata
            data["metadata"]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(self.watchlist_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Watchlist saved to {self.watchlist_file}")
        except Exception as e:
            logger.error(f"Error saving watchlist: {e}")
            raise
    
    def get_watchlist(self) -> List[str]:
        """Get current watchlist symbols
        
        Returns:
            List of watchlist symbols
        """
        data = self._load_watchlist()
        return data.get("watchlist", [])
    
    def add_symbol(self, symbol: str) -> bool:
        """Add symbol to watchlist
        
        Args:
            symbol: Stock symbol to add
            
        Returns:
            True if added, False if already exists
        """
        symbol = symbol.upper().strip()
        
        if not symbol:
            raise ValueError("Symbol cannot be empty")
        
        data = self._load_watchlist()
        watchlist = data.get("watchlist", [])
        
        if symbol in watchlist:
            logger.info(f"Symbol {symbol} already in watchlist")
            return False
        
        watchlist.append(symbol)
        data["watchlist"] = watchlist
        self._save_watchlist(data)
        
        logger.info(f"Added {symbol} to watchlist")
        return True
    
    def remove_symbol(self, symbol: str) -> bool:
        """Remove symbol from watchlist
        
        Args:
            symbol: Stock symbol to remove
            
        Returns:
            True if removed, False if not found
        """
        symbol = symbol.upper().strip()
        
        data = self._load_watchlist()
        watchlist = data.get("watchlist", [])
        
        if symbol not in watchlist:
            logger.info(f"Symbol {symbol} not found in watchlist")
            return False
        
        watchlist.remove(symbol)
        data["watchlist"] = watchlist
        self._save_watchlist(data)
        
        logger.info(f"Removed {symbol} from watchlist")
        return True
    
    def clear_watchlist(self):
        """Clear all symbols from watchlist"""
        data = self._load_watchlist()
        data["watchlist"] = []
        self._save_watchlist(data)
        logger.info("Cleared watchlist")
    
    def set_watchlist(self, symbols: List[str]):
        """Set entire watchlist
        
        Args:
            symbols: List of symbols to set as watchlist
        """
        # Clean and validate symbols
        clean_symbols = []
        for symbol in symbols:
            symbol = symbol.upper().strip()
            if symbol and symbol not in clean_symbols:
                clean_symbols.append(symbol)
        
        data = self._load_watchlist()
        data["watchlist"] = clean_symbols
        self._save_watchlist(data)
        
        logger.info(f"Set watchlist to {len(clean_symbols)} symbols")
    
    def get_watchlist_info(self) -> Dict:
        """Get full watchlist information including metadata
        
        Returns:
            Complete watchlist data including metadata
        """
        return self._load_watchlist()
    
    def is_symbol_in_watchlist(self, symbol: str) -> bool:
        """Check if symbol is in watchlist
        
        Args:
            symbol: Stock symbol to check
            
        Returns:
            True if symbol is in watchlist
        """
        symbol = symbol.upper().strip()
        watchlist = self.get_watchlist()
        return symbol in watchlist
    
    def get_watchlist_size(self) -> int:
        """Get number of symbols in watchlist
        
        Returns:
            Number of symbols in watchlist
        """
        return len(self.get_watchlist())
    
    def backup_watchlist(self, backup_file: Optional[str] = None) -> str:
        """Create backup of current watchlist
        
        Args:
            backup_file: Path for backup file. If None, creates timestamped backup.
            
        Returns:
            Path to backup file
        """
        if backup_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"watchlist_backup_{timestamp}.json"
        
        backup_path = Path(backup_file)
        
        # Copy current watchlist to backup
        data = self._load_watchlist()
        with open(backup_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Watchlist backed up to {backup_path}")
        return str(backup_path)
    
    def restore_from_backup(self, backup_file: str):
        """Restore watchlist from backup
        
        Args:
            backup_file: Path to backup file
        """
        backup_path = Path(backup_file)
        
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        try:
            with open(backup_path, 'r') as f:
                data = json.load(f)
            
            # Validate backup data
            if "watchlist" not in data:
                raise ValueError("Invalid backup file: missing watchlist")
            
            self._save_watchlist(data)
            logger.info(f"Watchlist restored from {backup_path}")
            
        except Exception as e:
            logger.error(f"Error restoring from backup: {e}")
            raise

# Convenience functions for global usage
_global_manager = None

def get_watchlist_manager() -> WatchlistManager:
    """Get global watchlist manager instance"""
    global _global_manager
    if _global_manager is None:
        _global_manager = WatchlistManager()
    return _global_manager

def get_default_watchlist() -> List[str]:
    """Get default watchlist symbols
    
    Returns:
        List of default watchlist symbols
    """
    manager = get_watchlist_manager()
    return manager.get_watchlist()

def add_to_watchlist(symbol: str) -> bool:
    """Add symbol to default watchlist
    
    Args:
        symbol: Stock symbol to add
        
    Returns:
        True if added, False if already exists
    """
    manager = get_watchlist_manager()
    return manager.add_symbol(symbol)

def remove_from_watchlist(symbol: str) -> bool:
    """Remove symbol from default watchlist
    
    Args:
        symbol: Stock symbol to remove
        
    Returns:
        True if removed, False if not found
    """
    manager = get_watchlist_manager()
    return manager.remove_symbol(symbol)