import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class Config:
    ALPACA_API_KEY: Optional[str] = os.getenv('ALPACA_API_KEY')
    ALPACA_SECRET_KEY: Optional[str] = os.getenv('ALPACA_SECRET_KEY')
    ALPACA_BASE_URL: str = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    
    FINNHUB_API_KEY: Optional[str] = os.getenv('FINNHUB_API_KEY')
    FRED_API_KEY: Optional[str] = os.getenv('FRED_API_KEY')
    COINDESK_API_URL: str = os.getenv('COINDESK_API_URL', 'https://api.coindesk.com/v1/bpi/currentprice.json')
    
    # LLM API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')
    OPENAI_BASE_URL: str = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    OPENAI_MODEL: str = os.getenv('OPENAI_MODEL', 'gpt-4-turbo-preview')
    
    MOONSHOT_API_KEY: Optional[str] = os.getenv('MOONSHOT_API_KEY')
    MOONSHOT_BASE_URL: str = os.getenv('MOONSHOT_BASE_URL', 'https://api.moonshot.cn/v1')
    MOONSHOT_MODEL: str = os.getenv('MOONSHOT_MODEL', 'moonshot-v1-8k')
    
    DEEPSEEK_API_KEY: Optional[str] = os.getenv('DEEPSEEK_API_KEY')
    DEEPSEEK_BASE_URL: str = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')
    DEEPSEEK_MODEL: str = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
    
    ANTHROPIC_API_KEY: Optional[str] = os.getenv('ANTHROPIC_API_KEY')
    ANTHROPIC_BASE_URL: str = os.getenv('ANTHROPIC_BASE_URL', 'https://api.anthropic.com')
    ANTHROPIC_MODEL: str = os.getenv('ANTHROPIC_MODEL', 'claude-3-sonnet-20240229')
    
    GEMINI_API_KEY: Optional[str] = os.getenv('GEMINI_API_KEY')
    GEMINI_MODEL: str = os.getenv('GEMINI_MODEL', 'gemini-pro')
    
    # Default LLM Provider
    DEFAULT_LLM_PROVIDER: str = os.getenv('DEFAULT_LLM_PROVIDER', 'openai')
    
    DATABASE_URL: str = os.getenv('DATABASE_URL', 'sqlite:///trading_system.db')
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    
    @classmethod
    def validate_alpaca_config(cls) -> bool:
        return bool(cls.ALPACA_API_KEY and cls.ALPACA_SECRET_KEY)
    
    @classmethod
    def validate_finnhub_config(cls) -> bool:
        return bool(cls.FINNHUB_API_KEY)
    
    @classmethod
    def validate_fred_config(cls) -> bool:
        return bool(cls.FRED_API_KEY)
    
    @classmethod
    def validate_openai_config(cls) -> bool:
        return bool(cls.OPENAI_API_KEY)
    
    @classmethod
    def validate_moonshot_config(cls) -> bool:
        return bool(cls.MOONSHOT_API_KEY)
    
    @classmethod
    def validate_deepseek_config(cls) -> bool:
        return bool(cls.DEEPSEEK_API_KEY)
    
    @classmethod
    def validate_anthropic_config(cls) -> bool:
        return bool(cls.ANTHROPIC_API_KEY)
    
    @classmethod
    def validate_gemini_config(cls) -> bool:
        return bool(cls.GEMINI_API_KEY)
    
    @classmethod
    def validate_llm_config(cls, provider: Optional[str] = None) -> bool:
        """Validate LLM configuration for specified provider or default"""
        provider = provider or cls.DEFAULT_LLM_PROVIDER
        
        validation_methods = {
            'openai': cls.validate_openai_config,
            'moonshot': cls.validate_moonshot_config,
            'deepseek': cls.validate_deepseek_config,
            'anthropic': cls.validate_anthropic_config,
            'claude': cls.validate_anthropic_config,  # Alias for Anthropic
            'gemini': cls.validate_gemini_config,
        }
        
        return validation_methods.get(provider.lower(), lambda: False)()
    
    @classmethod
    def get_llm_config(cls, provider: Optional[str] = None) -> dict:
        """Get LLM configuration for specified provider or default"""
        provider = provider or cls.DEFAULT_LLM_PROVIDER
        
        configs = {
            'openai': {
                'api_key': cls.OPENAI_API_KEY,
                'base_url': cls.OPENAI_BASE_URL,
                'model': cls.OPENAI_MODEL,
                'provider': 'openai'
            },
            'moonshot': {
                'api_key': cls.MOONSHOT_API_KEY,
                'base_url': cls.MOONSHOT_BASE_URL,
                'model': cls.MOONSHOT_MODEL,
                'provider': 'moonshot'
            },
            'deepseek': {
                'api_key': cls.DEEPSEEK_API_KEY,
                'base_url': cls.DEEPSEEK_BASE_URL,
                'model': cls.DEEPSEEK_MODEL,
                'provider': 'deepseek'
            },
            'anthropic': {
                'api_key': cls.ANTHROPIC_API_KEY,
                'base_url': cls.ANTHROPIC_BASE_URL,
                'model': cls.ANTHROPIC_MODEL,
                'provider': 'anthropic'
            },
            'claude': {  # Alias for Anthropic
                'api_key': cls.ANTHROPIC_API_KEY,
                'base_url': cls.ANTHROPIC_BASE_URL,
                'model': cls.ANTHROPIC_MODEL,
                'provider': 'anthropic'
            },
            'gemini': {
                'api_key': cls.GEMINI_API_KEY,
                'model': cls.GEMINI_MODEL,
                'provider': 'gemini'
            }
        }
        
        return configs.get(provider.lower(), configs['openai'])
    
    @classmethod
    def list_available_llm_providers(cls) -> list:
        """List all available LLM providers"""
        providers = []
        if cls.validate_openai_config():
            providers.append('openai')
        if cls.validate_moonshot_config():
            providers.append('moonshot')
        if cls.validate_deepseek_config():
            providers.append('deepseek')
        if cls.validate_anthropic_config():
            providers.append('anthropic')
        if cls.validate_gemini_config():
            providers.append('gemini')
        return providers