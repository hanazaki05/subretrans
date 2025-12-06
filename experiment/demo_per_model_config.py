#!/usr/bin/env python3
"""
Demo script showing per-model credential configuration with verbose output.

This demonstrates how different models can use different API endpoints
and keys, and how the -vvv verbose mode displays this information.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_sdk import ConfigSDK, MainModelSettings, TerminologyModelSettings
from llm_client_sdk import _resolve_model_credentials


def demo_scenario_1():
    """Scenario 1: Both models use global credentials."""
    print("\n" + "=" * 70)
    print("SCENARIO 1: Both models using global credentials")
    print("=" * 70)

    config = ConfigSDK(
        api_key="sk-global-key-example",
        api_base_url="https://api.openai.com/v1"
    )

    config.main_model = MainModelSettings(
        name="gpt-5-mini",
        max_output_tokens=27000,
        reasoning_effort="low",
        temperature=1.0
        # No key_file or base_url - uses global
    )

    config.terminology_model = TerminologyModelSettings(
        name="gpt-4o-mini",
        max_output_tokens=1800,
        temperature=0.45
        # No key_file or base_url - uses global
    )

    print("\nMain model credential resolution:")
    _resolve_model_credentials(config, config.main_model, verbose=True)

    print("Terminology model credential resolution:")
    _resolve_model_credentials(config, config.terminology_model, verbose=True)


def demo_scenario_2():
    """Scenario 2: Main model uses custom endpoint."""
    print("\n" + "=" * 70)
    print("SCENARIO 2: Main model with custom endpoint")
    print("=" * 70)

    config = ConfigSDK(
        api_key="sk-global-key-example",
        api_base_url="https://api.openai.com/v1"
    )

    config.main_model = MainModelSettings(
        name="gpt-5-mini",
        max_output_tokens=27000,
        reasoning_effort="low",
        temperature=1.0,
        base_url="https://my-custom-proxy.example.com/v1"  # Custom endpoint
    )

    config.terminology_model = TerminologyModelSettings(
        name="gpt-4o-mini",
        max_output_tokens=1800,
        temperature=0.45
        # Uses global
    )

    print("\nMain model credential resolution:")
    _resolve_model_credentials(config, config.main_model, verbose=True)

    print("Terminology model credential resolution:")
    _resolve_model_credentials(config, config.terminology_model, verbose=True)


def demo_scenario_3():
    """Scenario 3: Different endpoints for each model."""
    print("\n" + "=" * 70)
    print("SCENARIO 3: Different endpoints for each model")
    print("=" * 70)

    config = ConfigSDK(
        api_key="sk-global-key-example",
        api_base_url="https://api.openai.com/v1"
    )

    config.main_model = MainModelSettings(
        name="gpt-5-mini",
        max_output_tokens=27000,
        reasoning_effort="low",
        temperature=1.0,
        base_url="https://api.openai.com/v1"  # OpenAI for main model
    )

    config.terminology_model = TerminologyModelSettings(
        name="gpt-4o-mini",
        max_output_tokens=1800,
        temperature=0.45,
        base_url="http://localhost:8000/v1"  # Local server for terminology
    )

    print("\nMain model credential resolution:")
    _resolve_model_credentials(config, config.main_model, verbose=True)

    print("Terminology model credential resolution:")
    _resolve_model_credentials(config, config.terminology_model, verbose=True)


def demo_scenario_4():
    """Scenario 4: Different API keys for billing separation."""
    print("\n" + "=" * 70)
    print("SCENARIO 4: Different API keys for billing separation")
    print("=" * 70)
    print("\nNote: This demo shows how it would work with actual key files.")
    print("In real usage, key_file would point to actual files like '../key-main'")

    config = ConfigSDK(
        api_key="sk-default-key-example",
        api_base_url="https://api.openai.com/v1"
    )

    # In real usage, these would have key_file="../key-main", etc.
    config.main_model = MainModelSettings(
        name="gpt-5-mini",
        max_output_tokens=27000,
        reasoning_effort="low",
        temperature=1.0,
        # key_file="../key-main"  # Would use separate billing key
    )

    config.terminology_model = TerminologyModelSettings(
        name="gpt-4o-mini",
        max_output_tokens=1800,
        temperature=0.45,
        # key_file="../key-terminology"  # Would use separate billing key
    )

    print("\nMain model credential resolution (simulated):")
    print("  [Credential Resolution for gpt-5-mini]")
    print("    API Key: Model-specific (../key-main) [sk-proj-MainKey...]")
    print("    Base URL: Global (api.base_url) → https://api.openai.com/v1")
    print()

    print("Terminology model credential resolution (simulated):")
    print("  [Credential Resolution for gpt-4o-mini]")
    print("    API Key: Model-specific (../key-terminology) [sk-proj-TermKey...]")
    print("    Base URL: Global (api.base_url) → https://api.openai.com/v1")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PER-MODEL CREDENTIAL CONFIGURATION DEMO")
    print("=" * 70)
    print("\nThis demonstrates the new per-model credential override feature.")
    print("When running with -vvv flag, you'll see this output for each API call.")

    demo_scenario_1()
    demo_scenario_2()
    demo_scenario_3()
    demo_scenario_4()

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nTo use in production:")
    print("1. Edit experiment/config.yaml")
    print("2. Add key_file and/or base_url to main_model or terminology_model")
    print("3. Run with -vvv to see credential resolution:")
    print("   python experiment/main_sdk.py input.ass output.ass -vvv")
    print()
