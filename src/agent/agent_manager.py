"""
Agent Manager for Coordinating AI Analysis Tasks

Manages multiple AI agents and analysis workflows
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from .fundamental_analyzer import FundamentalAnalyzer
from .llm_providers import get_available_providers, create_provider

class AgentManager:
    """Manages AI agents and analysis workflows"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.analyzers = {}
        self.available_providers = get_available_providers()
        
        self.logger.info(f"Available LLM providers: {self.available_providers}")
    
    def get_analyzer(self, provider: str = 'openai') -> FundamentalAnalyzer:
        """Get or create fundamental analyzer for specific provider"""
        if provider not in self.analyzers:
            try:
                analyzer = FundamentalAnalyzer(
                    finnhub_key=self.config.get('finnhub_api_key'),
                    fred_key=self.config.get('fred_api_key'),
                    llm_provider=provider
                )
                self.analyzers[provider] = analyzer
                self.logger.info(f"Created analyzer with {provider} provider")
            except Exception as e:
                self.logger.error(f"Failed to create analyzer with {provider}: {e}")
                raise
        
        return self.analyzers[provider]
    
    def analyze_symbol(self, symbol: str, provider: str = 'openai') -> Dict[str, Any]:
        """Analyze a single symbol using specified provider"""
        try:
            analyzer = self.get_analyzer(provider)
            result = analyzer.analyze_company(symbol)
            
            self.logger.info(f"Completed analysis for {symbol} using {provider}")
            return result
            
        except Exception as e:
            self.logger.error(f"Analysis failed for {symbol} with {provider}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'provider': provider
            }
    
    def analyze_multiple_symbols(self, symbols: List[str], 
                                provider: str = 'openai') -> Dict[str, Dict[str, Any]]:
        """Analyze multiple symbols sequentially"""
        results = {}
        
        for symbol in symbols:
            self.logger.info(f"Analyzing {symbol}...")
            results[symbol] = self.analyze_symbol(symbol, provider)
        
        return results
    
    async def analyze_symbols_async(self, symbols: List[str], 
                                  provider: str = 'openai') -> Dict[str, Dict[str, Any]]:
        """Analyze multiple symbols asynchronously"""
        async def analyze_single(symbol: str) -> tuple:
            """Async wrapper for single symbol analysis"""
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self.analyze_symbol, symbol, provider
            )
            return symbol, result
        
        # Create tasks for all symbols
        tasks = [analyze_single(symbol) for symbol in symbols]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # Convert to dictionary
        return {symbol: result for symbol, result in results}
    
    def compare_providers(self, symbol: str, 
                         providers: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """Compare analysis from multiple LLM providers"""
        if providers is None:
            providers = self.available_providers
        
        if not providers:
            self.logger.warning("No LLM providers available")
            return {}
        
        results = {}
        
        for provider in providers:
            try:
                self.logger.info(f"Analyzing {symbol} with {provider}")
                results[provider] = self.analyze_symbol(symbol, provider)
            except Exception as e:
                self.logger.error(f"Failed to analyze {symbol} with {provider}: {e}")
                results[provider] = {
                    'error': str(e),
                    'symbol': symbol,
                    'provider': provider,
                    'timestamp': datetime.now().isoformat()
                }
        
        return results
    
    def generate_comparative_report(self, symbol: str, 
                                  provider_results: Dict[str, Dict[str, Any]]) -> str:
        """Generate comparative analysis report from multiple providers"""
        
        report = f"""
🌍 GAUSS WORLD TRADER - COMPARATIVE ANALYSIS REPORT
=================================================
Symbol: {symbol}
Generated: {datetime.now().isoformat()}
Providers Compared: {', '.join(provider_results.keys())}

"""
        
        for provider, result in provider_results.items():
            report += f"""
{provider.upper()} ANALYSIS:
{'-' * (len(provider) + 10)}
"""
            
            if 'error' in result:
                report += f"❌ Error: {result['error']}\n\n"
                continue
            
            # Financial grades
            financial = result.get('financial_analysis', {})
            grades = financial.get('ratio_grades', {})
            if grades:
                report += f"""
Financial Grades:
• Valuation: {grades.get('valuation', 'N/A')}
• Profitability: {grades.get('profitability', 'N/A')}
• Liquidity: {grades.get('liquidity', 'N/A')}
"""
            
            # News sentiment
            news = result.get('news_analysis', {})
            if news and 'error' not in news:
                report += f"• News Sentiment: {news.get('sentiment_label', 'N/A')}\n"
            
            # Analyst consensus
            analyst = result.get('analyst_analysis', {})
            if analyst:
                report += f"• Analyst Consensus: {analyst.get('consensus', 'N/A')}\n"
            
            # AI insights summary
            ai_insights = result.get('ai_insights')
            if ai_insights and isinstance(ai_insights, str) and 'error' not in ai_insights:
                # Extract first few sentences for summary
                sentences = ai_insights.split('.')[:3]
                summary = '. '.join(sentences) + '.' if sentences else ai_insights[:200] + '...'
                report += f"""
AI Insights Summary:
{summary}
"""
            
            report += "\n"
        
        # Consensus summary
        report += """
CONSENSUS SUMMARY:
-----------------
"""
        
        # Aggregate sentiment
        sentiments = []
        for result in provider_results.values():
            if 'error' not in result:
                news = result.get('news_analysis', {})
                if news and 'sentiment_label' in news:
                    sentiments.append(news['sentiment_label'])
        
        if sentiments:
            from collections import Counter
            sentiment_counts = Counter(sentiments)
            dominant_sentiment = sentiment_counts.most_common(1)[0][0]
            report += f"• Dominant News Sentiment: {dominant_sentiment}\n"
        
        # Aggregate grades
        all_grades = {}
        for result in provider_results.values():
            if 'error' not in result:
                grades = result.get('financial_analysis', {}).get('ratio_grades', {})
                for grade_type, grade in grades.items():
                    if grade_type not in all_grades:
                        all_grades[grade_type] = []
                    all_grades[grade_type].append(grade)
        
        for grade_type, grades in all_grades.items():
            if grades:
                from collections import Counter
                grade_counts = Counter(grades)
                dominant_grade = grade_counts.most_common(1)[0][0]
                report += f"• Consensus {grade_type.title()} Grade: {dominant_grade}\n"
        
        report += f"""

DISCLAIMER:
----------
This comparative analysis is for informational purposes only. Results may vary
between different AI providers due to their unique algorithms and training data.
Please conduct your own research and consult with a qualified financial advisor.

Generated by Gauss World Trader - Named after Carl Friedrich Gauss
Report Timestamp: {datetime.now().isoformat()}
"""
        
        return report
    
    def save_analysis(self, symbol: str, analysis_data: Dict[str, Any], 
                     filename: str = None) -> str:
        """Save analysis data to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fundamental_analysis_{symbol}_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=2, default=str)
            
            self.logger.info(f"Analysis saved to {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Failed to save analysis: {e}")
            raise
    
    def load_analysis(self, filename: str) -> Dict[str, Any]:
        """Load analysis data from file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.logger.info(f"Analysis loaded from {filename}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load analysis: {e}")
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and available resources"""
        status = {
            'available_llm_providers': self.available_providers,
            'active_analyzers': list(self.analyzers.keys()),
            'finnhub_configured': bool(self.config.get('finnhub_api_key')),
            'fred_configured': bool(self.config.get('fred_api_key')),
            'timestamp': datetime.now().isoformat()
        }
        
        return status