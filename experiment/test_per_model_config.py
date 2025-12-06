"""
Test per-model configuration overrides for API credentials.

This script tests that model-specific key_file and base_url settings
properly override global API settings.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_sdk import load_config_from_yaml, MainModelSettings, TerminologyModelSettings
from llm_client_sdk import _resolve_model_credentials


def test_global_credentials():
    """Test that global credentials are used when no model overrides exist."""
    print("Test 1: Global credentials (no model overrides)")
    print("-" * 60)

    # Load config with default values
    config = load_config_from_yaml()

    # Should use global settings
    api_key, base_url = _resolve_model_credentials(config, config.main_model, verbose=False)

    print(f"  Global API key: {config.api_key[:20]}...")
    print(f"  Global base URL: {config.api_base_url}")
    print(f"  Main model key_file: {config.main_model.key_file}")
    print(f"  Main model base_url: {config.main_model.base_url}")
    print(f"  Resolved API key: {api_key[:20]}...")
    print(f"  Resolved base URL: {base_url}")

    assert api_key == config.api_key, "Should use global API key"
    assert base_url == config.api_base_url, "Should use global base URL"
    print("  ✓ PASSED\n")


def test_model_specific_overrides():
    """Test that model-specific credentials override global settings."""
    print("Test 2: Model-specific credential overrides")
    print("-" * 60)

    # Create a config with model-specific overrides
    from config_sdk import ConfigSDK

    config = ConfigSDK(
        api_key="global-api-key-123",
        api_base_url="https://api.openai.com/v1"
    )

    # Create main model with overrides
    config.main_model = MainModelSettings(
        name="gpt-5-mini",
        max_output_tokens=12000,
        reasoning_effort="low",
        temperature=1.0,
        key_file=None,  # Will test base_url override only
        base_url="https://custom-endpoint.example.com/v1"
    )

    api_key, base_url = _resolve_model_credentials(config, config.main_model, verbose=False)

    print(f"  Global API key: {config.api_key}")
    print(f"  Global base URL: {config.api_base_url}")
    print(f"  Model base_url override: {config.main_model.base_url}")
    print(f"  Resolved API key: {api_key}")
    print(f"  Resolved base URL: {base_url}")

    assert api_key == "global-api-key-123", "Should use global API key when model key_file is None"
    assert base_url == "https://custom-endpoint.example.com/v1", "Should use model-specific base URL"
    print("  ✓ PASSED\n")


def test_different_models_different_credentials():
    """Test that different models can use different credentials."""
    print("Test 3: Different credentials for main vs terminology models")
    print("-" * 60)

    from config_sdk import ConfigSDK

    config = ConfigSDK(
        api_key="global-key",
        api_base_url="https://api.openai.com/v1"
    )

    # Main model with custom endpoint
    config.main_model = MainModelSettings(
        name="gpt-5-mini",
        max_output_tokens=12000,
        reasoning_effort="low",
        temperature=1.0,
        base_url="https://main-model-endpoint.com/v1"
    )

    # Terminology model with different endpoint
    config.terminology_model = TerminologyModelSettings(
        name="gpt-4o-mini",
        max_output_tokens=1800,
        temperature=0.5,
        base_url="https://terminology-endpoint.com/v1"
    )

    # Resolve for main model
    main_api_key, main_base_url = _resolve_model_credentials(config, config.main_model, verbose=False)

    # Resolve for terminology model
    term_api_key, term_base_url = _resolve_model_credentials(config, config.terminology_model, verbose=False)

    print(f"  Main model base URL: {main_base_url}")
    print(f"  Terminology model base URL: {term_base_url}")

    assert main_base_url == "https://main-model-endpoint.com/v1", "Main model should use its endpoint"
    assert term_base_url == "https://terminology-endpoint.com/v1", "Terminology model should use its endpoint"
    assert main_base_url != term_base_url, "Different models should use different endpoints"
    print("  ✓ PASSED\n")


def test_no_model_settings():
    """Test that global settings are used when model_settings is None."""
    print("Test 4: No model settings provided (fallback to global)")
    print("-" * 60)

    from config_sdk import ConfigSDK

    config = ConfigSDK(
        api_key="global-key",
        api_base_url="https://api.openai.com/v1"
    )

    # Resolve with no model settings
    api_key, base_url = _resolve_model_credentials(config, None, verbose=False)

    print(f"  Global API key: {config.api_key}")
    print(f"  Global base URL: {config.api_base_url}")
    print(f"  Resolved API key: {api_key}")
    print(f"  Resolved base URL: {base_url}")

    assert api_key == "global-key", "Should use global API key"
    assert base_url == "https://api.openai.com/v1", "Should use global base URL"
    print("  ✓ PASSED\n")


def test_verbose_output():
    """Test that verbose mode shows credential resolution info."""
    print("Test 5: Verbose credential resolution output (-vvv mode)")
    print("-" * 60)

    from config_sdk import ConfigSDK

    config = ConfigSDK(
        api_key="global-api-key-123",
        api_base_url="https://api.openai.com/v1"
    )

    # Create main model with overrides
    config.main_model = MainModelSettings(
        name="gpt-5-mini",
        max_output_tokens=12000,
        reasoning_effort="low",
        temperature=1.0,
        base_url="https://custom-endpoint.example.com/v1"
    )

    print("\n  With verbose=True, should display credential resolution:")
    api_key, base_url = _resolve_model_credentials(config, config.main_model, verbose=True)

    assert api_key == "global-api-key-123", "Should use global API key"
    assert base_url == "https://custom-endpoint.example.com/v1", "Should use model-specific base URL"
    print("  ✓ PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Per-Model Configuration Override Tests")
    print("=" * 60 + "\n")

    try:
        test_global_credentials()
        test_model_specific_overrides()
        test_different_models_different_credentials()
        test_no_model_settings()
        test_verbose_output()

        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
