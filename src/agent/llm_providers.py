"""
LLM Provider Interfaces for Multiple AI Services

Supports OpenAI, DeepSeek, Claude, Moonshot, and other providers
"""

import os
import requests
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import logging

class BaseLLMProvider(ABC):
    """Base class for all LLM providers"""
    
    def __init__(self, api_key: str, model: str = None):
        self.api_key = api_key
        self.model = model
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def generate_response(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Generate response from the LLM"""
        pass
    
    @abstractmethod
    def analyze_financial_data(self, data: Dict[str, Any]) -> str:
        """Analyze financial data and provide insights"""
        pass

class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        super().__init__(api_key or os.getenv('OPENAI_API_KEY'), model)
        self.base_url = "https://api.openai.com/v1"
    
    def generate_response(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Generate response using OpenAI API"""
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        messages = [{"role": "user", "content": prompt}]
        
        if context:
            system_message = f"Context: {json.dumps(context, indent=2)}"
            messages.insert(0, {"role": "system", "content": system_message})
        
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 2000,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            return f"Error generating response: {str(e)}"
    
    def analyze_financial_data(self, data: Dict[str, Any]) -> str:
        """Analyze financial data using OpenAI"""
        prompt = f"""
        As a financial analyst, analyze the following financial data and provide insights:
        
        {json.dumps(data, indent=2)}
        
        Please provide:
        1. Key financial metrics analysis
        2. Market sentiment assessment
        3. Risk factors identification
        4. Investment recommendations
        5. Technical and fundamental outlook
        
        Format your response as a structured analysis report.
        """
        
        return self.generate_response(prompt)

class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek AI provider"""
    
    def __init__(self, api_key: str = None, model: str = "deepseek-chat"):
        super().__init__(api_key or os.getenv('DEEPSEEK_API_KEY'), model)
        self.base_url = "https://api.deepseek.com/v1"
    
    def generate_response(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Generate response using DeepSeek API"""
        if not self.api_key:
            raise ValueError("DeepSeek API key not provided")
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        messages = [{"role": "user", "content": prompt}]
        
        if context:
            system_message = f"Context: {json.dumps(context, indent=2)}"
            messages.insert(0, {"role": "system", "content": system_message})
        
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 2000,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            self.logger.error(f"DeepSeek API error: {e}")
            return f"Error generating response: {str(e)}"
    
    def analyze_financial_data(self, data: Dict[str, Any]) -> str:
        """Analyze financial data using DeepSeek"""
        prompt = f"""
        Analyze the following financial data from a quantitative perspective:
        
        {json.dumps(data, indent=2)}
        
        Provide:
        1. Statistical analysis of key metrics
        2. Risk-return profile assessment
        3. Market efficiency indicators
        4. Quantitative trading signals
        5. Mathematical model recommendations
        
        Focus on data-driven insights and mathematical rigor.
        """
        
        return self.generate_response(prompt)

class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude provider"""
    
    def __init__(self, api_key: str = None, model: str = "claude-3-sonnet-20240229"):
        super().__init__(api_key or os.getenv('ANTHROPIC_API_KEY'), model)
        self.base_url = "https://api.anthropic.com/v1"
    
    def generate_response(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Generate response using Claude API"""
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")
        
        headers = {
            'x-api-key': self.api_key,
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01'
        }
        
        full_prompt = prompt
        if context:
            full_prompt = f"Context: {json.dumps(context, indent=2)}\n\n{prompt}"
        
        data = {
            "model": self.model,
            "max_tokens": 2000,
            "messages": [{"role": "user", "content": full_prompt}]
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result['content'][0]['text']
            
        except Exception as e:
            self.logger.error(f"Claude API error: {e}")
            return f"Error generating response: {str(e)}"
    
    def analyze_financial_data(self, data: Dict[str, Any]) -> str:
        """Analyze financial data using Claude"""
        prompt = f"""
        Conduct a comprehensive financial analysis of the following data:
        
        {json.dumps(data, indent=2)}
        
        Please provide:
        1. Fundamental analysis with key ratios
        2. Risk assessment and volatility analysis
        3. Market positioning and competitive analysis
        4. Economic factor considerations
        5. Strategic investment recommendations
        
        Provide balanced, nuanced insights with clear reasoning.
        """
        
        return self.generate_response(prompt)

class MoonshotProvider(BaseLLMProvider):
    """Moonshot AI provider"""
    
    def __init__(self, api_key: str = None, model: str = "moonshot-v1-8k"):
        super().__init__(api_key or os.getenv('MOONSHOT_API_KEY'), model)
        self.base_url = "https://api.moonshot.cn/v1"
    
    def generate_response(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Generate response using Moonshot API"""
        if not self.api_key:
            raise ValueError("Moonshot API key not provided")
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        messages = [{"role": "user", "content": prompt}]
        
        if context:
            system_message = f"Context: {json.dumps(context, indent=2)}"
            messages.insert(0, {"role": "system", "content": system_message})
        
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 2000,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            self.logger.error(f"Moonshot API error: {e}")
            return f"Error generating response: {str(e)}"
    
    def analyze_financial_data(self, data: Dict[str, Any]) -> str:
        """Analyze financial data using Moonshot"""
        prompt = f"""
        从中国市场角度分析以下金融数据：
        
        {json.dumps(data, indent=2)}
        
        请提供：
        1. 基本面分析和关键指标
        2. 市场情绪和投资者行为分析
        3. 政策影响和宏观经济因素
        4. 风险评估和投资建议
        5. 与中国市场的关联性分析
        
        请用中英文双语提供专业的金融分析报告。
        """
        
        return self.generate_response(prompt)

def get_available_providers() -> List[str]:
    """Get list of available LLM providers based on API keys"""
    providers = []
    
    if os.getenv('OPENAI_API_KEY'):
        providers.append('openai')
    if os.getenv('DEEPSEEK_API_KEY'):
        providers.append('deepseek')
    if os.getenv('ANTHROPIC_API_KEY'):
        providers.append('claude')
    if os.getenv('MOONSHOT_API_KEY'):
        providers.append('moonshot')
    
    return providers

def create_provider(provider_name: str, **kwargs) -> BaseLLMProvider:
    """Factory function to create LLM providers"""
    providers = {
        'openai': OpenAIProvider,
        'deepseek': DeepSeekProvider,
        'claude': ClaudeProvider,
        'moonshot': MoonshotProvider
    }
    
    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}")
    
    return providers[provider_name](**kwargs)