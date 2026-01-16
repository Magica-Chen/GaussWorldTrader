#!/usr/bin/env python3
"""
Watchlist Manager
Handles watchlist operations including reading, writing, adding, and removing symbols
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging

from src.utils.asset_utils import (
    infer_asset_type,
    normalize_asset_type,
    normalize_symbol,
)

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
                    {"symbol": "AAPL", "asset_type": "stock"},
                    {"symbol": "GOOGL", "asset_type": "stock"},
                    {"symbol": "MSFT", "asset_type": "stock"},
                    {"symbol": "TSLA", "asset_type": "stock"},
                    {"symbol": "NVDA", "asset_type": "stock"},
                    {"symbol": "AMZN", "asset_type": "stock"},
                    {"symbol": "META", "asset_type": "stock"},
                    {"symbol": "SPY", "asset_type": "stock"},
                    {"symbol": "QQQ", "asset_type": "stock"},
                    {"symbol": "VOO", "asset_type": "stock"},
                ],
                "metadata": {
                    "created": datetime.now().strftime("%Y-%m-%d"),
                    "last_updated": datetime.now().strftime("%Y-%m-%d"),
                    "description": "Gauss World Trader Default Watchlist",
                    "version": "2.0"
                }
            }
            
            try:
                with open(self.watchlist_file, 'w') as f:
                    json.dump(default_watchlist, f, indent=2)
                logger.info(f"Created default watchlist at {self.watchlist_file}")
            except Exception as e:
                logger.error(f"Error creating default watchlist: {e}")
                raise

    def _normalize_entry(self, entry: Any) -> Optional[Dict[str, str]]:
        if isinstance(entry, str):
            symbol = normalize_symbol(entry)
            if not symbol:
                return None
            return {"symbol": symbol, "asset_type": infer_asset_type(symbol)}
        if isinstance(entry, dict):
            symbol = entry.get("symbol") or entry.get("ticker") or entry.get("name")
            if not symbol:
                return None
            raw_asset_type = entry.get("asset_type")
            asset_type = (
                normalize_asset_type(raw_asset_type)
                if raw_asset_type
                else infer_asset_type(symbol)
            )
            normalized_symbol = normalize_symbol(symbol, asset_type)
            if not normalized_symbol:
                return None
            return {"symbol": normalized_symbol, "asset_type": asset_type}
        return None

    def _normalize_watchlist_entries(self, entries: Any) -> List[Dict[str, str]]:
        normalized: List[Dict[str, str]] = []
        seen = set()
        for entry in entries or []:
            normalized_entry = self._normalize_entry(entry)
            if not normalized_entry:
                continue
            key = (normalized_entry["symbol"], normalized_entry["asset_type"])
            if key in seen:
                continue
            seen.add(key)
            normalized.append(normalized_entry)
        return normalized
    
    def _load_watchlist(self) -> Dict:
        """Load watchlist from JSON file"""
        try:
            with open(self.watchlist_file, 'r') as f:
                data = json.load(f)
                data["watchlist"] = self._normalize_watchlist_entries(data.get("watchlist", []))
                if "metadata" not in data:
                    data["metadata"] = {
                        "created": datetime.now().strftime("%Y-%m-%d"),
                        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "description": "Gauss World Trader Default Watchlist",
                        "version": "2.0",
                    }
                return data
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
            data["watchlist"] = self._normalize_watchlist_entries(data.get("watchlist", []))
            # Update metadata
            if "metadata" not in data:
                data["metadata"] = {
                    "created": datetime.now().strftime("%Y-%m-%d"),
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "description": "Gauss World Trader Default Watchlist",
                    "version": "2.0",
                }
            else:
                data["metadata"]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(self.watchlist_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Watchlist saved to {self.watchlist_file}")
        except Exception as e:
            logger.error(f"Error saving watchlist: {e}")
            raise
    
    def get_watchlist_entries(self, asset_type: Optional[str] = None) -> List[Dict[str, str]]:
        """Get current watchlist entries.

        Args:
            asset_type: Optional asset type filter (stock, crypto, option)

        Returns:
            List of watchlist entries with symbol and asset_type
        """
        data = self._load_watchlist()
        entries = data.get("watchlist", [])
        if asset_type:
            normalized_type = normalize_asset_type(asset_type)
            entries = [entry for entry in entries if entry.get("asset_type") == normalized_type]
        return entries

    def get_watchlist(self, asset_type: Optional[str] = None) -> List[str]:
        """Get current watchlist symbols
        
        Returns:
            List of watchlist symbols
        """
        entries = self.get_watchlist_entries(asset_type)
        return [entry["symbol"] for entry in entries]
    
    def add_symbol(self, symbol: str, asset_type: str = "stock") -> bool:
        """Add symbol to watchlist
        
        Args:
            symbol: Stock symbol to add
            asset_type: Asset type (stock, crypto, option)
            
        Returns:
            True if added, False if already exists
        """
        normalized_type = normalize_asset_type(asset_type)
        symbol = normalize_symbol(symbol, normalized_type)
        
        if not symbol:
            raise ValueError("Symbol cannot be empty")
        
        data = self._load_watchlist()
        watchlist = data.get("watchlist", [])
        entry = {"symbol": symbol, "asset_type": normalized_type}
        
        if any(item["symbol"] == symbol and item["asset_type"] == normalized_type for item in watchlist):
            logger.info(f"Symbol {symbol} ({normalized_type}) already in watchlist")
            return False
        
        watchlist.append(entry)
        data["watchlist"] = watchlist
        self._save_watchlist(data)
        
        logger.info(f"Added {symbol} ({normalized_type}) to watchlist")
        return True
    
    def remove_symbol(self, symbol: str, asset_type: Optional[str] = None) -> bool:
        """Remove symbol from watchlist
        
        Args:
            symbol: Stock symbol to remove
            asset_type: Optional asset type filter
            
        Returns:
            True if removed, False if not found
        """
        normalized_type = normalize_asset_type(asset_type) if asset_type else None
        symbol = normalize_symbol(symbol, normalized_type)
        
        data = self._load_watchlist()
        watchlist = data.get("watchlist", [])
        
        if normalized_type:
            remaining = [
                item for item in watchlist
                if not (item["symbol"] == symbol and item["asset_type"] == normalized_type)
            ]
        else:
            remaining = [item for item in watchlist if item["symbol"] != symbol]

        if len(remaining) == len(watchlist):
            logger.info(f"Symbol {symbol} not found in watchlist")
            return False
        
        data["watchlist"] = remaining
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
        entries = self._normalize_watchlist_entries(symbols)
        
        data = self._load_watchlist()
        data["watchlist"] = entries
        self._save_watchlist(data)
        
        logger.info(f"Set watchlist to {len(entries)} symbols")
    
    def get_watchlist_info(self) -> Dict:
        """Get full watchlist information including metadata
        
        Returns:
            Complete watchlist data including metadata
        """
        return self._load_watchlist()
    
    def is_symbol_in_watchlist(self, symbol: str, asset_type: Optional[str] = None) -> bool:
        """Check if symbol is in watchlist
        
        Args:
            symbol: Stock symbol to check
            asset_type: Optional asset type filter
            
        Returns:
            True if symbol is in watchlist
        """
        symbol = normalize_symbol(symbol, asset_type)
        entries = self.get_watchlist_entries(asset_type)
        return any(entry["symbol"] == symbol for entry in entries)
    
    def get_watchlist_size(self, asset_type: Optional[str] = None) -> int:
        """Get number of symbols in watchlist
        
        Returns:
            Number of symbols in watchlist
        """
        return len(self.get_watchlist_entries(asset_type))
    
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

def get_default_watchlist(asset_type: Optional[str] = None) -> List[str]:
    """Get default watchlist symbols
    
    Returns:
        List of default watchlist symbols
    """
    manager = get_watchlist_manager()
    return manager.get_watchlist(asset_type=asset_type)

def add_to_watchlist(symbol: str, asset_type: str = "stock") -> bool:
    """Add symbol to default watchlist
    
    Args:
        symbol: Stock symbol to add
        asset_type: Asset type (stock, crypto, option)
        
    Returns:
        True if added, False if already exists
    """
    manager = get_watchlist_manager()
    return manager.add_symbol(symbol, asset_type=asset_type)

def remove_from_watchlist(symbol: str, asset_type: Optional[str] = None) -> bool:
    """Remove symbol from default watchlist
    
    Args:
        symbol: Stock symbol to remove
        asset_type: Optional asset type filter
        
    Returns:
        True if removed, False if not found
    """
    manager = get_watchlist_manager()
    return manager.remove_symbol(symbol, asset_type=asset_type)
