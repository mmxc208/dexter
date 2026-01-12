"""Financial tools for Dexter Strands agent.

This module provides a centralized registry of all financial tools available to the agent.
Tools are organized by category for easy discovery and maintenance.

Total tools: 15
- Fundamentals: 4 tools (income statements, balance sheets, cash flow, all statements)
- Filings: 4 tools (filing metadata, 10-K items, 10-Q items, 8-K items)
- Prices: 2 tools (snapshot, historical)
- Metrics: 2 tools (snapshot, historical)
- News: 1 tool
- Estimates: 1 tool
- Segments: 1 tool
"""

# Import all tools from finance package
from dexter_strands.tools.finance import (
    # Fundamentals (4 tools)
    get_income_statements,
    get_balance_sheets,
    get_cash_flow_statements,
    get_all_financial_statements,
    # Filings (4 tools)
    get_filings,
    get_10K_filing_items,
    get_10Q_filing_items,
    get_8K_filing_items,
    # Prices (2 tools)
    get_price_snapshot,
    get_prices,
    # Metrics (2 tools)
    get_financial_metrics_snapshot,
    get_financial_metrics,
    # News (1 tool)
    get_news,
    # Estimates (1 tool)
    get_analyst_estimates,
    # Segments (1 tool)
    get_segmented_revenues,
)

# Flat list of all tools for agent initialization
ALL_TOOLS = [
    # Fundamentals
    get_income_statements,
    get_balance_sheets,
    get_cash_flow_statements,
    get_all_financial_statements,
    # Filings
    get_filings,
    get_10K_filing_items,
    get_10Q_filing_items,
    get_8K_filing_items,
    # Prices
    get_price_snapshot,
    get_prices,
    # Metrics
    get_financial_metrics_snapshot,
    get_financial_metrics,
    # News
    get_news,
    # Estimates
    get_analyst_estimates,
    # Segments
    get_segmented_revenues,
]

# Categorized tools for future use (e.g., selective tool loading, documentation)
TOOLS_BY_CATEGORY = {
    "fundamentals": [
        get_income_statements,
        get_balance_sheets,
        get_cash_flow_statements,
        get_all_financial_statements,
    ],
    "filings": [
        get_filings,
        get_10K_filing_items,
        get_10Q_filing_items,
        get_8K_filing_items,
    ],
    "prices": [
        get_price_snapshot,
        get_prices,
    ],
    "metrics": [
        get_financial_metrics_snapshot,
        get_financial_metrics,
    ],
    "news": [
        get_news,
    ],
    "estimates": [
        get_analyst_estimates,
    ],
    "segments": [
        get_segmented_revenues,
    ],
}

__all__ = [
    "ALL_TOOLS",
    "TOOLS_BY_CATEGORY",
    # Individual tools
    "get_income_statements",
    "get_balance_sheets",
    "get_cash_flow_statements",
    "get_all_financial_statements",
    "get_filings",
    "get_10K_filing_items",
    "get_10Q_filing_items",
    "get_8K_filing_items",
    "get_price_snapshot",
    "get_prices",
    "get_financial_metrics_snapshot",
    "get_financial_metrics",
    "get_news",
    "get_analyst_estimates",
    "get_segmented_revenues",
]
