"""
Fundamental Analysis Engine with AI Integration

Combines financial data with AI analysis to generate comprehensive reports
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
import json

from src.data.finnhub_provider import FinnhubProvider
from src.data.fred_provider import FREDProvider
from .llm_providers import create_provider, get_available_providers

class FundamentalAnalyzer:
    """Comprehensive fundamental analysis with AI insights"""
    
    def __init__(self, 
                 finnhub_key: str = None, 
                 fred_key: str = None,
                 llm_provider: str = 'openai',
                 llm_model: str = None):
        
        self.finnhub = FinnhubProvider(finnhub_key)
        self.fred = FREDProvider(fred_key)
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM provider
        try:
            self.llm = create_provider(llm_provider, model=llm_model)
            self.llm_available = True
        except Exception as e:
            self.logger.warning(f"Could not initialize LLM provider {llm_provider}: {e}")
            self.llm = None
            self.llm_available = False
    
    def analyze_company(self, symbol: str) -> Dict[str, Any]:
        """Comprehensive company analysis"""
        self.logger.info(f"Starting fundamental analysis for {symbol}")
        
        # Gather all data
        market_data = self._get_comprehensive_market_data(symbol)
        
        # Perform financial ratio analysis
        financial_analysis = self._analyze_financial_ratios(market_data.get('basic_financials', {}))
        
        # Analyze insider information
        insider_analysis = self._analyze_insider_data(
            market_data.get('insider_transactions', []),
            market_data.get('insider_sentiment', {})
        )
        
        # Economic context analysis
        economic_analysis = self._analyze_economic_context(market_data.get('economic_indicators', {}))
        
        # Analyst recommendations analysis
        analyst_analysis = self._analyze_analyst_recommendations(
            market_data.get('recommendations', {}),
            market_data.get('price_target', {})
        )
        
        # Compile comprehensive report
        analysis_result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'company_profile': market_data.get('company_profile', {}),
            'financial_analysis': financial_analysis,
            'insider_analysis': insider_analysis,
            'economic_analysis': economic_analysis,
            'analyst_analysis': analyst_analysis,
            'raw_data': market_data
        }
        
        # Generate AI insights if available
        if self.llm_available:
            try:
                ai_insights = self._generate_ai_insights(analysis_result)
                analysis_result['ai_insights'] = ai_insights
            except Exception as e:
                self.logger.error(f"Error generating AI insights: {e}")
                analysis_result['ai_insights'] = {"error": str(e)}
        
        return analysis_result

    def _get_comprehensive_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive market data from multiple sources."""
        data: Dict[str, Any] = {}

        # Finnhub data
        data['company_profile'] = self.finnhub.get_company_profile(symbol)
        data['basic_financials'] = self.finnhub.get_basic_financials(symbol)
        data['company_news'] = self.finnhub.get_company_news(symbol)
        data['recommendations'] = self.finnhub.get_recommendation_trends(symbol)
        data['price_target'] = self.finnhub.get_price_target(symbol)
        data['quote'] = self.finnhub.get_quote(symbol)
        data['earnings_surprises'] = self.finnhub.get_earnings_surprises(symbol)
        data['insider_transactions'] = self.finnhub.get_insider_transactions(symbol)
        data['insider_sentiment'] = self.finnhub.get_insider_sentiment(symbol)

        # FRED economic data
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        data['economic_indicators'] = self.fred.get_economic_indicators(start_date)

        return data
    
    def _analyze_financial_ratios(self, financials: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze key financial ratios"""
        if not financials or 'metric' not in financials:
            return {"error": "No financial data available"}
        
        metrics = financials.get('metric', {})
        
        analysis = {
            'valuation_ratios': {
                'pe_ratio': metrics.get('peBasicExclExtraTTM'),
                'pb_ratio': metrics.get('pbQuarterly'),
                'ps_ratio': metrics.get('psQuarterly'),
                'ev_ebitda': metrics.get('evEbitdaTTM'),
                'peg_ratio': metrics.get('pegRatioTTM')
            },
            'profitability_ratios': {
                'roe': metrics.get('roeRfy'),
                'roa': metrics.get('roaRfy'),
                'gross_margin': metrics.get('grossMarginTTM'),
                'operating_margin': metrics.get('operatingMarginTTM'),
                'net_margin': metrics.get('netProfitMarginTTM')
            },
            'liquidity_ratios': {
                'current_ratio': metrics.get('currentRatioQuarterly'),
                'quick_ratio': metrics.get('quickRatioQuarterly'),
                'cash_ratio': metrics.get('cashRatioQuarterly')
            },
            'leverage_ratios': {
                'debt_to_equity': metrics.get('totalDebtToEquityQuarterly'),
                'debt_to_assets': metrics.get('totalDebtToTotalCapitalQuarterly'),
                'interest_coverage': metrics.get('interestCoverageQuarterly')
            },
            'efficiency_ratios': {
                'asset_turnover': metrics.get('assetTurnoverTTM'),
                'inventory_turnover': metrics.get('inventoryTurnoverTTM'),
                'receivables_turnover': metrics.get('receivablesTurnoverTTM')
            }
        }
        
        # Calculate ratio grades
        analysis['ratio_grades'] = self._grade_financial_ratios(analysis)
        
        return analysis
    
    def _grade_financial_ratios(self, ratios: Dict[str, Any]) -> Dict[str, str]:
        """Grade financial ratios"""
        grades = {}
        
        # Valuation grades
        pe_ratio = ratios['valuation_ratios'].get('pe_ratio')
        if pe_ratio:
            if pe_ratio < 15:
                grades['valuation'] = 'A'
            elif pe_ratio < 25:
                grades['valuation'] = 'B'
            elif pe_ratio < 35:
                grades['valuation'] = 'C'
            else:
                grades['valuation'] = 'D'
        
        # Profitability grades
        roe = ratios['profitability_ratios'].get('roe')
        if roe:
            if roe > 0.15:
                grades['profitability'] = 'A'
            elif roe > 0.10:
                grades['profitability'] = 'B'
            elif roe > 0.05:
                grades['profitability'] = 'C'
            else:
                grades['profitability'] = 'D'
        
        # Liquidity grades
        current_ratio = ratios['liquidity_ratios'].get('current_ratio')
        if current_ratio:
            if current_ratio > 2.0:
                grades['liquidity'] = 'A'
            elif current_ratio > 1.5:
                grades['liquidity'] = 'B'
            elif current_ratio > 1.0:
                grades['liquidity'] = 'C'
            else:
                grades['liquidity'] = 'D'
        
        return grades
    
    def _analyze_insider_data(self, insider_transactions: List[Dict[str, Any]], 
                             insider_sentiment: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze insider transactions and sentiment"""
        analysis = {}
        
        # Analyze insider transactions
        if insider_transactions and not any('error' in item for item in insider_transactions if isinstance(item, dict)):
            transactions_analysis = {
                'total_transactions': len(insider_transactions),
                'recent_count': len([t for t in insider_transactions[:10]]),
                'net_change': 0,
                'buy_transactions': 0,
                'sell_transactions': 0,
                'insider_activity_level': 'Low'
            }
            
            # Calculate aggregated metrics
            for transaction in insider_transactions[:20]:  # Recent 20 transactions
                change = transaction.get('change', 0)
                if isinstance(change, (int, float)):
                    transactions_analysis['net_change'] += change
                    
                    # Count buy/sell based on change sign
                    if change > 0:
                        transactions_analysis['buy_transactions'] += 1
                    elif change < 0:
                        transactions_analysis['sell_transactions'] += 1
            
            # Determine activity level
            total_recent = transactions_analysis['recent_count']
            if total_recent > 15:
                transactions_analysis['insider_activity_level'] = 'High'
            elif total_recent > 5:
                transactions_analysis['insider_activity_level'] = 'Moderate'
            
            # Determine sentiment from net change
            net_change = transactions_analysis['net_change']
            if net_change > 50000:
                transactions_analysis['transaction_sentiment'] = 'Bullish'
            elif net_change < -50000:
                transactions_analysis['transaction_sentiment'] = 'Bearish'
            else:
                transactions_analysis['transaction_sentiment'] = 'Neutral'
            
            analysis['transactions'] = transactions_analysis
        else:
            analysis['transactions'] = {"error": "No insider transaction data available"}
        
        # Analyze insider sentiment
        if insider_sentiment and 'data' in insider_sentiment:
            sentiment_data = insider_sentiment['data']
            if sentiment_data:
                latest_data = sentiment_data[-1] if sentiment_data else {}
                
                sentiment_analysis = {
                    'latest_mspr': latest_data.get('mspr', 0),
                    'latest_change': latest_data.get('change', 0),
                    'latest_period': f"{latest_data.get('year', 'N/A')}-{latest_data.get('month', 'N/A'):02d}",
                    'data_points': len(sentiment_data)
                }
                
                # Interpret MSPR (Monthly Share Purchase Ratio)
                mspr = sentiment_analysis['latest_mspr']
                if mspr > 0.5:
                    sentiment_analysis['mspr_interpretation'] = 'Bullish (High insider buying)'
                elif mspr < -0.5:
                    sentiment_analysis['mspr_interpretation'] = 'Bearish (High insider selling)'
                else:
                    sentiment_analysis['mspr_interpretation'] = 'Neutral (Balanced activity)'
                
                analysis['sentiment'] = sentiment_analysis
            else:
                analysis['sentiment'] = {"error": "No insider sentiment data points"}
        else:
            analysis['sentiment'] = {"error": "No insider sentiment data available"}
        
        return analysis
    
    def _analyze_economic_context(self, economic_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze economic context"""
        if not economic_data:
            return {"error": "No economic data available"}
        
        analysis = {}
        
        for indicator, data in economic_data.items():
            if data.empty or 'error' in data.columns:
                continue
            
            latest_value = data['value'].iloc[-1] if not data.empty else None
            previous_value = data['value'].iloc[-2] if len(data) > 1 else None
            
            analysis[indicator] = {
                'latest_value': latest_value,
                'previous_value': previous_value,
                'change': latest_value - previous_value if latest_value and previous_value else None,
                'trend': 'Rising' if latest_value and previous_value and latest_value > previous_value else 'Falling'
            }
        
        # Economic environment assessment
        fed_rate = analysis.get('Federal_Funds_Rate', {}).get('latest_value', 0)
        unemployment = analysis.get('Unemployment', {}).get('latest_value', 0)
        
        if fed_rate and unemployment:
            if fed_rate < 2 and unemployment < 5:
                analysis['economic_environment'] = 'Accommodative'
            elif fed_rate > 4 or unemployment > 7:
                analysis['economic_environment'] = 'Challenging'
            else:
                analysis['economic_environment'] = 'Neutral'
        
        return analysis
    
    def _analyze_analyst_recommendations(self, recommendations: Dict[str, Any], 
                                       price_target: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze analyst recommendations"""
        analysis = {}
        
        if recommendations and 'error' not in recommendations:
            recent_rec = recommendations[0] if recommendations else {}
            analysis['recommendations'] = {
                'strong_buy': recent_rec.get('strongBuy', 0),
                'buy': recent_rec.get('buy', 0),
                'hold': recent_rec.get('hold', 0),
                'sell': recent_rec.get('sell', 0),
                'strong_sell': recent_rec.get('strongSell', 0),
                'period': recent_rec.get('period', 'N/A')
            }
            
            # Calculate consensus
            total = sum([
                recent_rec.get('strongBuy', 0),
                recent_rec.get('buy', 0),
                recent_rec.get('hold', 0),
                recent_rec.get('sell', 0),
                recent_rec.get('strongSell', 0)
            ])
            
            if total > 0:
                buy_ratio = (recent_rec.get('strongBuy', 0) + recent_rec.get('buy', 0)) / total
                if buy_ratio > 0.6:
                    analysis['consensus'] = 'Strong Buy'
                elif buy_ratio > 0.4:
                    analysis['consensus'] = 'Buy'
                elif buy_ratio > 0.2:
                    analysis['consensus'] = 'Hold'
                else:
                    analysis['consensus'] = 'Sell'
        
        if price_target and 'error' not in price_target:
            analysis['price_target'] = {
                'target_high': price_target.get('targetHigh'),
                'target_low': price_target.get('targetLow'),
                'target_mean': price_target.get('targetMean'),
                'target_median': price_target.get('targetMedian'),
                'last_updated': price_target.get('lastUpdated')
            }
        
        return analysis
    
    def _generate_ai_insights(self, analysis_data: Dict[str, Any]) -> str:
        """Generate AI-powered insights"""
        if not self.llm:
            return "AI insights not available"
        
        # Prepare data for AI analysis
        summary_data = {
            'symbol': analysis_data['symbol'],
            'financial_grades': analysis_data['financial_analysis'].get('ratio_grades', {}),
            'insider_sentiment': analysis_data['insider_analysis'].get('sentiment', {}).get('mspr_interpretation', 'Unknown'),
            'economic_environment': analysis_data['economic_analysis'].get('economic_environment', 'Unknown'),
            'analyst_consensus': analysis_data['analyst_analysis'].get('consensus', 'Unknown'),
            'valuation_ratios': analysis_data['financial_analysis'].get('valuation_ratios', {}),
            'profitability_ratios': analysis_data['financial_analysis'].get('profitability_ratios', {})
        }
        
        return self.llm.analyze_financial_data(summary_data)
    
    def generate_report(self, analysis_data: Dict[str, Any]) -> str:
        """Generate comprehensive fundamental analysis report"""
        symbol = analysis_data['symbol']
        timestamp = analysis_data['timestamp']
        
        report = f"""
🌍 GAUSS WORLD TRADER - FUNDAMENTAL ANALYSIS REPORT
==================================================
Symbol: {symbol}
Generated: {timestamp}

COMPANY OVERVIEW:
----------------
"""
        
        company_profile = analysis_data.get('company_profile', {})
        if company_profile and 'error' not in company_profile:
            report += f"""
• Name: {company_profile.get('name', 'N/A')}
• Industry: {company_profile.get('finnhubIndustry', 'N/A')}
• Market Cap: ${company_profile.get('marketCapitalization', 0):,.0f}M
• Country: {company_profile.get('country', 'N/A')}
• Website: {company_profile.get('weburl', 'N/A')}
"""
        
        # Financial Analysis Section
        financial = analysis_data.get('financial_analysis', {})
        if financial and 'error' not in financial:
            report += """
FINANCIAL ANALYSIS:
------------------
"""
            grades = financial.get('ratio_grades', {})
            if grades:
                report += f"""
• Valuation Grade: {grades.get('valuation', 'N/A')}
• Profitability Grade: {grades.get('profitability', 'N/A')}
• Liquidity Grade: {grades.get('liquidity', 'N/A')}
"""
            
            valuation = financial.get('valuation_ratios', {})
            if valuation:
                report += f"""
Valuation Ratios:
• P/E Ratio: {valuation.get('pe_ratio', 'N/A')}
• P/B Ratio: {valuation.get('pb_ratio', 'N/A')}
• EV/EBITDA: {valuation.get('ev_ebitda', 'N/A')}
"""
        
        # Insider Analysis
        insider = analysis_data.get('insider_analysis', {})
        if insider:
            report += f"""
INSIDER ANALYSIS:
-----------------"""
            
            # Transactions analysis
            transactions = insider.get('transactions', {})
            if transactions and 'error' not in transactions:
                report += f"""
• Recent Transactions: {transactions.get('recent_count', 0)}
• Net Share Change: {transactions.get('net_change', 0):+,.0f}
• Activity Level: {transactions.get('insider_activity_level', 'Unknown')}
• Transaction Sentiment: {transactions.get('transaction_sentiment', 'Neutral')}
"""
            
            # Sentiment analysis
            sentiment = insider.get('sentiment', {})
            if sentiment and 'error' not in sentiment:
                report += f"""
• Latest MSPR: {sentiment.get('latest_mspr', 0):.2f}
• MSPR Interpretation: {sentiment.get('mspr_interpretation', 'Unknown')}
• Data Period: {sentiment.get('latest_period', 'N/A')}
"""
        
        # Economic Context
        economic = analysis_data.get('economic_analysis', {})
        if economic and 'error' not in economic:
            report += f"""
ECONOMIC ENVIRONMENT:
--------------------
• Assessment: {economic.get('economic_environment', 'Unknown')}
"""
            
            if 'Federal_Funds_Rate' in economic:
                fed_data = economic['Federal_Funds_Rate']
                report += f"• Federal Funds Rate: {fed_data.get('latest_value', 'N/A')}%\n"
            
            if 'Unemployment' in economic:
                unemployment_data = economic['Unemployment']
                report += f"• Unemployment Rate: {unemployment_data.get('latest_value', 'N/A')}%\n"
        
        # Analyst Recommendations
        analyst = analysis_data.get('analyst_analysis', {})
        if analyst:
            report += f"""
ANALYST RECOMMENDATIONS:
-----------------------
• Consensus: {analyst.get('consensus', 'N/A')}
"""
            
            price_target = analyst.get('price_target', {})
            if price_target:
                report += f"""
• Target Mean: ${price_target.get('target_mean', 'N/A')}
• Target High: ${price_target.get('target_high', 'N/A')}
• Target Low: ${price_target.get('target_low', 'N/A')}
"""
        
        # AI Insights
        ai_insights = analysis_data.get('ai_insights')
        if ai_insights and isinstance(ai_insights, str) and 'error' not in ai_insights:
            report += f"""
AI-POWERED INSIGHTS:
-------------------
{ai_insights}
"""
        
        report += f"""
DISCLAIMER:
----------
This analysis is for informational purposes only and should not be considered
as investment advice. Please conduct your own research and consult with a
qualified financial advisor before making investment decisions.

Generated by Gauss World Trader - Named after Carl Friedrich Gauss
Report Timestamp: {timestamp}
"""
        
        return report
