#!/usr/bin/env python3
"""
Simple test script for the Text2SQL API
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing /health...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_generate_no_retrieval():
    """Test generate without retrieval"""
    print("\nTesting /generate (no retrieval)...")
    try:
        data = {
            "question": "Show all singers",
            "db_id": "concert_singer",
            "use_retrieval": False
        }
        response = requests.post(
            f"{BASE_URL}/generate",
            json=data,
            timeout=60
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_generate_with_retrieval():
    """Test generate with retrieval"""
    print("\nTesting /generate (with retrieval)...")
    try:
        data = {
            "question": "Show all singers from USA",
            "db_id": "concert_singer",
            "use_retrieval": True,
            "top_k": 5
        }
        response = requests.post(
            f"{BASE_URL}/generate",
            json=data,
            timeout=60
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Question: {result['question']}")
            print(f"Generated SQL: {result['generated_sql']}")
            print(f"Retrieved {len(result.get('retrieved_context', []))} context entries")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Text2SQL API Test Suite")
    print("=" * 60)
    
    # Wait a moment for server to be ready
    print("\nWaiting for server...")
    time.sleep(2)
    
    # Run tests
    results = []
    results.append(("Health Check", test_health()))
    results.append(("Generate (No Retrieval)", test_generate_no_retrieval()))
    results.append(("Generate (With Retrieval)", test_generate_with_retrieval()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")
