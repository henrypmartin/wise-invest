'''
Created on 24-Nov-2024

@author: Henry Martin
'''
import logging
from bs4 import BeautifulSoup
import aiohttp
from typing import Dict, List, Any, Union, Optional
import asyncio
from langchain.schema import SystemMessage, HumanMessage
import json
import re
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

class DynamicPortfolioAllocator:    
    def __init__(self, llm):
        # Set up logging
        self.llm = llm
        self.sources = {
            'moneycontrol': 'https://www.moneycontrol.com/stocksmarketsindia/',
            'nseindia': 'https://www.nseindia.com/',
            'economic_times': 'https://economictimes.indiatimes.com/markets'
        }

    def format_amount(self, amount: Optional[float]) -> str:
        """Format amount in Indian currency format"""
        if amount is None:
            return "Not specified"
        try:
                if amount >= 10000000:  # 1 crore
                    return f"₹{amount/10000000:.2f} Cr"
                elif amount >= 100000:  # 1 lakh
                    return f"₹{amount/100000:.2f} L"
                return f"₹{amount:,.2f}"
        except Exception as e:
                logger.error(f"Error formatting amount: {e}")
                return "Not specified"

    async def get_market_sentiment(self) -> Dict[str, Any]:
        """Fetch current market sentiment from financial websites"""
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_page(session, url) for url in self.sources.values()]
            pages = await asyncio.gather(*tasks)

            sentiment_data = self._extract_sentiment(pages)
            return sentiment_data

    async def get_dynamic_allocation(self, query: str, risk_profile: str = None, investment_amount: float = None) -> Dict:
        """Generate dynamic allocation using market data and LLM"""
        try:
            market_sentiment = await self.get_market_sentiment()

            system_message = SystemMessage(content="""You are an expert Indian financial advisor.
            Analyze market conditions and user profile to recommend an optimal portfolio allocation
            across different asset classes. Consider both traditional and alternative investments
            based on the current market scenario and risk profile. Take into account the user specific requirements""")

            human_message = HumanMessage(content=f"""
            Create a comprehensive portfolio allocation based on:

            User Query: {query}

            Additional Context:
            - Risk Profile: {risk_profile if risk_profile else 'To be determined from query'}
            - Investment Amount: ₹{self.format_amount(investment_amount) if investment_amount else 'Not specified'}

            Market Context:
            {json.dumps(market_sentiment, indent=2)}

            First, analyze the user's query to understand:
            1. Their specific requirements
            2. Investment goals
            3. Risk tolerance (if not explicitly provided)
            4. Investment horizon
            5. Any constraints or preferences

            Consider these factors:
            1. Current market conditions and trends
            2. Asset class valuations and momentum
            3. Risk-adjusted returns potential
            4. Market volatility levels
            5. Economic indicators and outlook
            6. Liquidity requirements
            7. Current market prices of instruments
            8. Diversification benefits
            9. Correlation between asset classes
            10. Investment horizon implications

            Return a JSON with this structure:
            Then provide recommendations in this JSON structure:
            {{
                "query_analysis": {{
                    "understood_requirements": "<list key points from query>",
                    "investment_goals": "<identified goals>",
                    "risk_profile": "<assessed risk profile>",
                    "investment_horizon": "<estimated timeline>",
                    "special_considerations": "<any specific requirements>"
                }},

                "portfolio_strategy": {{
                    "recommended_asset_classes": [
                        {{
                            "name": "<asset_class_name>",
                            "allocation": <float 0-1>,
                            "rationale": "<why this allocation>",
                            "total_investment": <amount in INR>,
                            "instruments": [
                                {{
                                    "type": "<instrument_type>",
                                    "symbol": "<market_symbol>",
                                    "name": "<instrument_name>",
                                    "weight": <float 0-1>,
                                    "current_price": <price in INR>,
                                    "units_to_buy": <integer>,
                                    "total_investment": <amount in INR>,
                                    "rationale": "<brief rationale>",
                                    "key_metrics": {{
                                        "yield": "<if applicable>",
                                        "risk_measure": "<e.g., beta, volatility>",
                                        "liquidity_score": "<high/medium/low>",
                                        "additional_metrics": {{}}
                                    }}
                                }}
                            ]
                        }}
                    ]
                }},
                "investment_rationale": [
                    {{
                        "factor": "<decision factor>",
                        "impact": "<impact description>",
                        "recommendation": "<specific recommendation>"
                    }}
                ],
                "risk_analysis": {{
                    "portfolio_level_risks": [
                        {{
                            "risk_type": "<type of risk>",
                            "severity": "<high/medium/low>",
                            "mitigation_strategy": "<how to mitigate>"
                        }}
                    ],
                    "asset_class_specific_risks": {{
                        "<asset_class>": [
                            {{
                                "risk_factor": "<specific risk>",
                                "impact": "<potential impact>",
                                "mitigation": "<mitigation approach>"
                            }}
                        ]
                    }}
                }},
                "portfolio_metrics": {{
                    "expected_returns": {{
                        "1_year": "<percentage>",
                        "3_year": "<percentage>",
                        "5_year": "<percentage>",
                        "10_year": "<percentage>"
                    }},
                    "risk_metrics": {{
                        "portfolio_beta": <float>,
                        "volatility": "<percentage>",
                        "sharpe_ratio": <float>,
                        "max_drawdown": "<percentage>"
                    }},
                    "diversification_metrics": {{
                        "concentration_score": <float>,
                        "correlation_score": <float>
                    }},
                    "portfolio_summary": {{
                        "total_investment": <amount in INR>,
                        "cash_remaining": <amount in INR>,
                        "number_of_instruments": <integer>,
                        "rebalancing_frequency": "<recommendation>"
                    }}
                }}
            }}

            Requirements:
            1. Recommend appropriate asset classes based on market conditions and risk profile
            2. All allocations must sum to 1.0
            3. Use only valid market symbols
            4. Include current market prices
            5. Calculate exact number of units based on price
            6. Consider minimum investment requirements
            7. Provide detailed rationale for each major decision
            8. Include comprehensive risk analysis
            9. Ensure total investments match the given amount
            10. Provide forward-looking metrics and rebalancing guidance

            Note: Feel free to add any additional asset classes or instruments that you believe
            would benefit the portfolio given the current market conditions and risk profile.
            """)

            response = await self._get_llm_response(system_message, human_message)
            allocation = json.loads(self._extract_json(response))
            return allocation

        except Exception as e:
            logger.error(f"Error generating allocation: {e}")
            raise

    async def _fetch_page(self, session: aiohttp.ClientSession, url: str) -> str:
        """Fetch webpage content"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        try:
            async with session.get(url, headers=headers) as response:
                return await response.text()
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return ""

    def _extract_sentiment(self, pages: List[str]) -> Dict[str, Any]:
        """Extract market sentiment from webpages"""
        sentiment_data = {
            'market_mood': 'neutral',
            'trending_sectors': [],
            'key_indicators': {},
            'risk_metrics': {}
        }

        for page in pages:
            soup = BeautifulSoup(page, 'html.parser')
            # Extract relevant data points
            # Add your specific extraction logic here

        return sentiment_data

    async def _get_llm_response(self, system_message: SystemMessage,
                              human_message: HumanMessage) -> str:
        """Get LLM response with error handling"""
        try:
            response = await asyncio.to_thread(
                lambda: self.llm.invoke([system_message, human_message])
            )
            return response.content
        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            raise

    def _extract_json(self, content: str) -> str:
        """Extract JSON from LLM response"""
        
        try:
            # Find JSON block between triple backticks
            json_match = re.search(r'\{[\s\S]*\}', content)
            if not json_match:
                raise ValueError("No valid JSON found in response")
            return json_match.group()
        except Exception as e:
            logger.error(f"Error extracting JSON: {e}")
            return "{}"

def format_portfolio_response(allocation: Dict, original_query : str) -> str:    
    """Format dynamic portfolio allocation into a detailed, readable report"""

    def format_amount(amount: float) -> str:
        """Format currency in Indian format"""
        try:
            if amount >= 10000000:  # 1 crore
                return f"₹{amount/10000000:.2f} Cr"
            elif amount >= 100000:  # 1 lakh
                return f"₹{amount/100000:.2f} L"
            return f"₹{amount:,.2f}"
        except (TypeError, ValueError):
            return "N/A"

    def format_pct(value: Union[float, str]) -> str:
        """Format percentage values"""
        try:
            if isinstance(value, str) and '%' in value:
                return value
            return f"{float(value)*100:.1f}%"
        except (TypeError, ValueError):
            return "N/A"

    def format_instrument_details(instrument: Dict) -> List[str]:
        """Format instrument details with metrics"""
        details = [
            f"  • {instrument.get('name', 'Unknown')} ({instrument.get('symbol', 'N/A')})",
            f"    Price: {format_amount(instrument.get('current_price', 0))} | "
            f"Units: {instrument.get('units_to_buy', 0):,} | "
            f"Total: {format_amount(instrument.get('total_investment', 0))}"
        ]

        # Add key metrics if available
        if 'key_metrics' in instrument:
            metrics = instrument['key_metrics']
            metric_lines = []
            if metrics.get('yield'):
                metric_lines.append(f"Yield: {metrics['yield']}")
            if metrics.get('risk_measure'):
                metric_lines.append(f"Risk: {metrics['risk_measure']}")
            if metrics.get('liquidity_score'):
                metric_lines.append(f"Liquidity: {metrics['liquidity_score']}")
            if metric_lines:
                details.append(f"    Metrics: {' | '.join(metric_lines)}")

        # Add rationale if available
        if 'rationale' in instrument:
            details.append(f"    Rationale: {instrument['rationale']}")

        return details

    try:
        response = []


            # Query Analysis Section
        if 'query_analysis' in allocation:
            query = allocation['query_analysis']
            response.extend([
                "Query Analysis",
                "=" * 40,
                f"Original Query: {original_query}",
                f"Understood Requirements: {query.get('understood_requirements', 'N/A')}",
                f"Investment Goals: {query.get('investment_goals', 'N/A')}",
                f"Risk Profile: {query.get('risk_profile', 'N/A')}",
                f"Investment Horizon: {query.get('investment_horizon', 'N/A')}",
                ""
            ])
        # Portfolio Header with Summary Metrics
        response.extend([
            "Investment Portfolio Strategy",
            "=" * 60,
            ""
        ])

        # Portfolio Metrics Summary
        if 'portfolio_metrics' in allocation:
            metrics = allocation['portfolio_metrics']
            summary = metrics.get('portfolio_summary', {})

            response.extend([
                "Portfolio Summary",
                "-" * 20,
                f"Total Investment: {format_amount(summary.get('total_investment', 0))}",
                f"Cash Remaining: {format_amount(summary.get('cash_remaining', 0))}",
                f"Number of Instruments: {summary.get('number_of_instruments', 'N/A')}",
                f"Recommended Rebalancing: {summary.get('rebalancing_frequency', 'N/A')}",
                ""
            ])

            # Expected Returns
            if 'expected_returns' in metrics:
                returns = metrics['expected_returns']
                response.extend([
                    "Expected Returns",
                    "-" * 20,
                    f"1 Year: {returns.get('1_year', 'N/A')}",
                    f"3 Year: {returns.get('3_year', 'N/A')}",
                    f"5 Year: {returns.get('5_year', 'N/A')}",
                    ""
                ])

            # Risk Metrics
            if 'risk_metrics' in metrics:
                risk = metrics['risk_metrics']
                response.extend([
                    "Risk Metrics",
                    "-" * 20,
                    f"Portfolio Beta: {risk.get('portfolio_beta', 'N/A')}",
                    f"Volatility: {risk.get('volatility', 'N/A')}",
                    f"Sharpe Ratio: {risk.get('sharpe_ratio', 'N/A')}",
                    f"Maximum Drawdown: {risk.get('max_drawdown', 'N/A')}",
                    ""
                ])

        # Asset Allocation and Instruments
        if 'portfolio_strategy' in allocation and 'recommended_asset_classes' in allocation['portfolio_strategy']:
            response.extend([
                "Asset Allocation Strategy",
                "=" * 40
            ])

            for asset_class in allocation['portfolio_strategy']['recommended_asset_classes']:
                response.extend([
                    f"\n{asset_class['name']}",
                    f"Allocation: {format_pct(asset_class['allocation'])}",
                    f"Investment: {format_amount(asset_class.get('total_investment', 0))}",
                    f"Rationale: {asset_class.get('rationale', 'N/A')}",
                    "\nInstruments:"
                ])

                for instrument in asset_class.get('instruments', []):
                    response.extend(format_instrument_details(instrument))
                response.append("")

        # Investment Rationale
        if 'investment_rationale' in allocation:
            response.extend([
                "Investment Rationale",
                "=" * 40
            ])
            for rationale in allocation['investment_rationale']:
                response.extend([
                    f"\nFactor: {rationale.get('factor', 'N/A')}",
                    f"Impact: {rationale.get('impact', 'N/A')}",
                    f"Recommendation: {rationale.get('recommendation', 'N/A')}"
                ])
            response.append("")

        # Risk Analysis
        if 'risk_analysis' in allocation:
            risk_analysis = allocation['risk_analysis']

            # Portfolio Level Risks
            if 'portfolio_level_risks' in risk_analysis:
                response.extend([
                    "Portfolio Risk Analysis",
                    "=" * 40
                ])
                for risk in risk_analysis['portfolio_level_risks']:
                    response.extend([
                        f"\nRisk Type: {risk.get('risk_type', 'N/A')}",
                        f"Severity: {risk.get('severity', 'N/A')}",
                        f"Mitigation: {risk.get('mitigation_strategy', 'N/A')}"
                    ])
                response.append("")

            # Asset Class Specific Risks
            if 'asset_class_specific_risks' in risk_analysis:
                response.extend([
                    "Asset-Specific Risks",
                    "=" * 40
                ])
                for asset_class, risks in risk_analysis['asset_class_specific_risks'].items():
                    response.append(f"\n{asset_class}:")
                    for risk in risks:
                        response.extend([
                            f"  • Risk: {risk.get('risk_factor', 'N/A')}",
                            f"    Impact: {risk.get('impact', 'N/A')}",
                            f"    Mitigation: {risk.get('mitigation', 'N/A')}"
                        ])
                response.append("")


        # Implementation Steps
        response.extend([
            "Implementation Strategy",
            "=" * 40,
            "1. Review the complete portfolio strategy",
            "2. Verify risk tolerance alignment",
            "3. Set up necessary trading accounts",
            "4. Execute trades in order of priority",
            "5. Set up systematic investment plans where applicable",
            "6. Implement suggested risk mitigation strategies",
            "7. Schedule regular portfolio reviews",
            ""
        ])

        # Disclaimer
        response.extend([
            "Disclaimer",
            "=" * 40,
            "This is an AI-generated investment recommendation based on current market data.",
            "All investments are subject to market risks. Please read all scheme-related documents carefully.",
            "Past performance is not indicative of future returns.",
            "Consult with a SEBI registered investment advisor before making investment decisions.",
            ""
        ])

        return "\n".join(response)

    except Exception as e:
        return (f"Error formatting portfolio response: {str(e)}\n"
                f"Raw allocation: {json.dumps(allocation, indent=2)}")

llm = ChatOpenAI(temperature=0.2, model="gpt-4o")
allocator = DynamicPortfolioAllocator(llm)

async def run_portfolio_allocator_details(query, risk_profile, investment_amount):
    logger.info(f"Inside run_portfolio_allocator with args: {query}; {risk_profile}; {investment_amount}")
    allocation = await allocator.get_dynamic_allocation(query=query, risk_profile=risk_profile, investment_amount=investment_amount)
    
    return format_portfolio_response(allocation, query)

#query='I want to invest 5 lakhs in Indian market with aggressive risk, can you suggest the stocks and the amount of shares to purchase'

#det = asyncio.run(run_portfolio_allocator(query, 'aggressive', 500000))

#print(det)
    