"""
Timezone utilities for trading operations
Centralizes timezone handling for the entire application
"""
import pytz
from datetime import datetime, time

# Global timezone for all trading operations
EASTERN = pytz.timezone('US/Eastern')

def get_market_hours():
    """Get standard market hours in ET"""
    return {
        'market_open': time(9, 30),  # 9:30 AM ET
        'market_close': time(16, 0),  # 4:00 PM ET
        'premarket_start': time(4, 0),  # 4:00 AM ET
        'postmarket_end': time(20, 0)  # 8:00 PM ET
    }

def get_market_status(current_time: datetime = None) -> str:
    """
    Determine current market status
    Returns: 'closed', 'open', 'pre-market', 'post-market'
    """
    if current_time is None:
        current_time = datetime.now(EASTERN)
    elif current_time.tzinfo is None:
        current_time = EASTERN.localize(current_time)
    else:
        current_time = current_time.astimezone(EASTERN)
    
    # Check if it's a weekend
    if current_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return 'closed'
    
    current_time_only = current_time.time()
    market_hours = get_market_hours()
    
    if current_time_only < market_hours['premarket_start']:
        return 'closed'
    elif current_time_only < market_hours['market_open']:
        return 'pre-market'
    elif current_time_only <= market_hours['market_close']:
        return 'open'
    elif current_time_only <= market_hours['postmarket_end']:
        return 'post-market'
    else:
        return 'closed'

def now_et() -> datetime:
    """Get current time in Eastern timezone"""
    return datetime.now(EASTERN)

def to_et(dt: datetime) -> datetime:
    """Convert datetime to Eastern timezone"""
    if dt.tzinfo is None:
        return EASTERN.localize(dt)
    else:
        return dt.astimezone(EASTERN)


def format_duration(seconds: float) -> str:
    """Format seconds as Hh Mm Ss."""
    total = int(max(0, round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}h {minutes}m {secs}s"
