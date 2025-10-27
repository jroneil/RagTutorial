#!/usr/bin/env python3
"""
Test script to check Ollama server connectivity
"""

import requests
import time

def test_ollama_connection(base_url):
    """Test connection to Ollama server"""
    try:
        print(f"Testing connection to {base_url}...")
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        if response.status_code == 200:
            print(f"SUCCESS: Connected to Ollama server at {base_url}")
            models = response.json().get('models', [])
            if models:
                print("Available models:")
                for model in models:
                    print(f"  - {model.get('name', 'Unknown')}")
            else:
                print("No models found")
            return True
        else:
            print(f"FAILED: Server returned status code {response.status_code}")
            return False
    except requests.exceptions.ConnectTimeout:
        print(f"FAILED: Connection timeout to {base_url}")
        return False
    except requests.exceptions.ConnectionError:
        print(f"FAILED: Could not connect to {base_url}")
        return False
    except Exception as e:
        print(f"FAILED: Unexpected error: {e}")
        return False

def main():
    # Test the configured URL
    configured_url = "http://192.168.1.235:11434"
    test_ollama_connection(configured_url)
    
    print("\n" + "="*50)
    print("Testing common local Ollama URLs...")
    
    # Test common local URLs
    local_urls = [
        "http://localhost:11434",
        "http://127.0.0.1:11434",
        "http://0.0.0.0:11434"
    ]
    
    for url in local_urls:
        if test_ollama_connection(url):
            print(f"\nSUGGESTION: Update your rag_app.py to use: {url}")
            break
        time.sleep(1)  # Small delay between tests

if __name__ == "__main__":
    main()
