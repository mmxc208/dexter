# Dexter-Strands Documentation

This directory contains comprehensive documentation for the Strands SDK migration project, including setup instructions, usage examples, and learning notes from Phase 1 implementation.

## Quick Start

### Prerequisites

- Python 3.10 or higher
- `uv` package manager ([installation guide](https://github.com/astral-sh/uv))
- AWS account with Bedrock access
- Financial Datasets API key

### Installation

```bash
# Clone the repository
cd dexter-strands

# Install dependencies with uv
uv sync

# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# Required: AWS_REGION, BEDROCK_MODEL_ID, FINANCIAL_DATASETS_API_KEY, etc.
```

### Running Dexter

```bash
# Start the CLI
uv run dexter-strands

# Or with explicit path
uv run python -m dexter_strands.cli
```

## Usage Examples

### Basic Queries

```
>> What is Apple's revenue for the last quarter?
⠋ Thinking...

──────────────────────────────────────────────────────────────────────
Based on Apple's most recent quarterly income statement (Q4 2024):

Revenue: $94.93 billion
- This represents a 6% increase year-over-year
- iPhone revenue: $46.22 billion
- Services revenue: $24.97 billion
...
──────────────────────────────────────────────────────────────────────

>> Show me Tesla's balance sheet
>> Compare Microsoft and Google's profit margins
>> exit
👋 Goodbye! Thanks for using Dexter.
```

### Available Tools

Dexter has access to three financial data tools:

1. **get_income_statement** - Revenue, expenses, net income
2. **get_balance_sheet** - Assets, liabilities, equity
3. **get_stock_price** - Current price, volume, OHLC data

## Configuration

### Required Environment Variables

Create a `.env` file with the following variables:

```bash
# AWS Bedrock Configuration
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0
MODEL_TEMPERATURE=0.0

# AWS Credentials (optional if using IAM role or SSO)
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here

# Financial Datasets API
FINANCIAL_DATASETS_API_KEY=your_api_key_here
FINANCIAL_API_BASE_URL=https://api.financialdatasets.ai

# Agent Settings
MAX_ITERATIONS=20
MAX_STEPS_PER_TASK=5

# Logging
LOG_LEVEL=INFO
LOG_FILE=dexter-strands.log
```

See `.env.example` for a complete template with descriptions.

### AWS Authentication

Dexter supports multiple AWS authentication methods:

1. **Explicit credentials** (from `.env` file)
2. **AWS SSO** (recommended for development)
3. **AWS credentials file** (`~/.aws/credentials`)
4. **IAM role** (when running on EC2/ECS/Lambda)

For AWS SSO:
```bash
aws sso login --profile your-profile
export AWS_PROFILE=your-profile
# Leave AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY empty in .env
```

## Architecture Overview

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI Interface                            │
│                  (dexter_strands/cli.py)                    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                  Strands Agent                              │
│              (dexter_strands/agent.py)                      │
│  - Agent Builder pattern                                    │
│  - Tool registration                                        │
│  - Conversation manager                                     │
│  - Safety limits                                            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              AWS Bedrock Model                              │
│            (dexter_strands/model.py)                        │
│  - Claude 4.5 Haiku                                         │
│  - Cross-region inference                                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              Financial Tools                                │
│          (dexter_strands/tools/finance.py)                  │
└─────────────────────────────────────────────────────────────┘
```

### Key Features

- **Autonomous Tool Selection**: Agent automatically selects and executes appropriate tools
- **Conversation Management**: Maintains context across multiple queries
- **Safety Limits**: Prevents infinite loops with iteration limits and loop detection
- **Error Handling**: Graceful error handling with user-friendly messages
- **Structured Logging**: Logs to both console and file for debugging

## Testing

Run the test suite:

```bash
cd dexter-strands
.venv/bin/pytest

# Run with coverage
.venv/bin/pytest --cov=dexter_strands

# Run specific test file
.venv/bin/pytest tests/test_agent.py
```

All 113 tests should pass.

## Troubleshooting

### Configuration Errors

If you see configuration errors on startup:
1. Check that all required variables are set in `.env`
2. Verify API keys are valid
3. Ensure AWS credentials are configured correctly

### AWS Bedrock Access Denied

If you see "You don't have access to the model" errors:
1. Verify you have enabled model access in AWS Bedrock console
2. Check your IAM permissions include `bedrock:InvokeModel`
3. Ensure your region supports the model (use global profile for best coverage)

### SSO Token Expiration

If using AWS SSO and requests start failing:
```bash
aws sso login --profile your-profile
```

## Purpose

These documents serve as:
1. **Learning Resources**: Understanding Strands SDK patterns and concepts
2. **Migration Guides**: Converting from LangChain to Strands
3. **Reference Material**: Quick lookup for implementation details
4. **Knowledge Base**: Team documentation and onboarding

## Phase 1 Focus

Phase 1 documentation focuses on:
- Basic agent creation and configuration
- AWS Bedrock integration with Claude 4.5 Haiku
- Tool definition and registration
- Conversation management with sliding window
- Safety limits and error handling
- Simple CLI interface

- **Phase 2**: Complete tool migration (15+ financial tools)
- **Phase 3**: Rich UI with streaming responses
- **Phase 4**: Context management and offloading
- **Phase 5**: Multi-agent orchestration (Swarm/Graph patterns)
- **Phase 6**: Tracing and observability

## Future Phases

Future phases will expand on:
- **Phase 7**: Evaluation framework

## Contributing

When adding new features or documentation:
1. Update relevant documentation files
2. Add inline code comments for Strands-specific patterns
3. Include examples and usage instructions
4. Update this README if adding new documentation files

## Resources

- [Strands SDK Documentation](https://github.com/awslabs/strands-agents-sdk-python)
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Financial Datasets API](https://financialdatasets.ai/)
- [uv Package Manager](https://github.com/astral-sh/uv)
