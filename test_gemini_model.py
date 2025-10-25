#!/usr/bin/env python3
"""
Test Gemini API model availability
Run this locally to verify the model name works
"""
import os
import sys
from google import genai
from google.genai import types

def test_model(model_name: str, api_key: str):
    """Test if a Gemini model is available and working"""
    print(f"\n{'='*60}")
    print(f"Testing model: {model_name}")
    print(f"{'='*60}")

    try:
        # Initialize client
        client = genai.Client(api_key=api_key)
        print("✅ Client initialized")

        # Test generateContent
        response = client.models.generate_content(
            model=model_name,
            contents="Say 'Hello' if you can hear me",
            config=types.GenerateContentConfig(
                temperature=0.0,
            )
        )

        # Check response
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                text = ""
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        text += part.text

                print(f"✅ Model responded: {text[:100]}")
                return True

        print("❌ No response from model")
        return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def list_available_models(api_key: str):
    """List all available Gemini models"""
    print(f"\n{'='*60}")
    print("Available Gemini Models")
    print(f"{'='*60}")

    try:
        client = genai.Client(api_key=api_key)

        models = list(client.models.list())

        print(f"\nTotal models: {len(models)}")
        print("\nModels that support generateContent:")

        for model in models:
            # Check if model supports generateContent
            if hasattr(model, 'supported_generation_methods'):
                if 'generateContent' in model.supported_generation_methods:
                    print(f"  ✅ {model.name}")
            else:
                # Fallback: just list all models
                print(f"  • {model.name}")

        return True

    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return False

def main():
    # Get API key from environment
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        print("❌ ERROR: GEMINI_API_KEY environment variable not set")
        print("\nUsage:")
        print("  export GEMINI_API_KEY='your-api-key-here'")
        print("  python test_gemini_model.py")
        sys.exit(1)

    print("="*60)
    print("Gemini API Model Test")
    print("="*60)

    # List available models
    list_available_models(api_key)

    # Test old (broken) model name
    print("\n\n" + "="*60)
    print("Testing OLD model name (should FAIL)")
    print("="*60)
    test_model("gemini-2.5-pro-002", api_key)

    # Test new (correct) model name
    print("\n\n" + "="*60)
    print("Testing NEW model name (should SUCCEED)")
    print("="*60)
    success = test_model("gemini-2.5-pro", api_key)

    if success:
        print("\n" + "="*60)
        print("✅ SUCCESS: gemini-2.5-pro is working!")
        print("="*60)
        print("\nYou can now update agent_v5/agent.py:")
        print("  Line 97: model=\"gemini-2.5-pro\",")
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("❌ FAILED: gemini-2.5-pro not working")
        print("="*60)
        print("\nCheck your API key and try again")
        sys.exit(1)

if __name__ == "__main__":
    main()
