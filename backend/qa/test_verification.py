"""Test script to verify the RAG verification system is working."""

import json
import requests
import sys
from typing import Dict, Any

# Default API URL (adjust if needed)
# API_URL = "http://localhost:5000"  # Local
API_URL = "http://localhost:8000"  # Docker


def test_verification_enabled():
    """Test that verification is enabled and returns results."""
    print("=" * 60)
    print("Test 1: Verification Enabled")
    print("=" * 60)
    
    query = {
        "query": "What was Apple's revenue in 2023?",
        "filters": {"ticker": "AAPL", "year": "2023"},
        "top_k": 10,
        "enable_verification": True
    }
    
    try:
        response = requests.post(f"{API_URL}/query", json=query, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        # Check if verification is present
        if "verification" in data:
            verification = data["verification"]
            print("✓ Verification results found!")
            print(f"\nVerification Details:")
            
            # Safely format scores
            def format_score(score):
                if score is None:
                    return "N/A"
                try:
                    return f"{float(score):.2f}"
                except (ValueError, TypeError):
                    return str(score)
            
            print(f"  Overall Score: {format_score(verification.get('overall_score'))}")
            print(f"  Answer-Source Alignment: {format_score(verification.get('answer_source_alignment'))}")
            print(f"  Citation Coverage: {format_score(verification.get('citation_coverage'))}")
            print(f"  Fact Verification Score: {format_score(verification.get('fact_verification_score'))}")
            
            issues = verification.get('issues', [])
            if issues:
                print(f"\n  Issues Found: {len(issues)}")
                for issue in issues:
                    print(f"    - {issue}")
            else:
                print("\n  ✓ No issues detected")
            
            verified_sources = verification.get('verified_sources', [])
            print(f"\n  Verified Sources: {len(verified_sources)}")
            if verified_sources:
                print(f"    Source IDs: {verified_sources[:5]}...")  # Show first 5
            
            unverified_claims = verification.get('unverified_claims', [])
            if unverified_claims:
                print(f"\n  Unverified Claims: {len(unverified_claims)}")
                for claim in unverified_claims[:3]:  # Show first 3
                    print(f"    - {claim[:100]}...")
            else:
                print("\n  ✓ All claims verified")
            
            return True
        else:
            print("✗ Verification results NOT found in response")
            print("  Response keys:", list(data.keys()))
            print("\n  Full response (first 500 chars):")
            print(f"  {str(data)[:500]}...")
            print("\n  Possible reasons:")
            print("    - Verification failed silently (check service logs)")
            print("    - enable_verification is False in config")
            print("    - Verification threw an exception")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                print(f"  Response status: {e.response.status_code}")
                print(f"  Response body: {e.response.text[:200]}")
            except:
                pass
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        print(f"  Traceback: {traceback.format_exc()}")
        return False


def test_verification_disabled():
    """Test that verification can be disabled."""
    print("\n" + "=" * 60)
    print("Test 2: Verification Disabled")
    print("=" * 60)
    
    query = {
        "query": "What was Apple's revenue in 2023?",
        "filters": {"ticker": "AAPL", "year": "2023"},
        "top_k": 10,
        "enable_verification": False
    }
    
    try:
        response = requests.post(f"{API_URL}/query", json=query, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        if "verification" not in data:
            print("✓ Verification correctly disabled (not in response)")
            return True
        else:
            print("✗ Verification should not be present when disabled")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                print(f"  Response status: {e.response.status_code}")
                print(f"  Response body: {e.response.text[:200]}")
            except:
                pass
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        print(f"  Traceback: {traceback.format_exc()}")
        return False


def test_verification_scores():
    """Test that verification scores are in valid range."""
    print("\n" + "=" * 60)
    print("Test 3: Verification Score Validation")
    print("=" * 60)
    
    query = {
        "query": "What was Apple's revenue in 2023?",
        "filters": {"ticker": "AAPL", "year": "2023"},
        "top_k": 10
    }
    
    try:
        response = requests.post(f"{API_URL}/query", json=query, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        if "verification" not in data:
            print("✗ Verification not found")
            return False
        
        verification = data["verification"]
        scores = {
            "overall_score": verification.get("overall_score"),
            "answer_source_alignment": verification.get("answer_source_alignment"),
            "citation_coverage": verification.get("citation_coverage"),
            "fact_verification_score": verification.get("fact_verification_score")
        }
        
        all_valid = True
        for name, score in scores.items():
            if score is None:
                print(f"✗ {name} is missing (None)")
                all_valid = False
            else:
                try:
                    score_float = float(score)
                    if not (0.0 <= score_float <= 1.0):
                        print(f"✗ {name} = {score_float} is not in range [0.0, 1.0]")
                        all_valid = False
                    else:
                        print(f"✓ {name} = {score_float:.2f} (valid)")
                except (ValueError, TypeError) as e:
                    print(f"✗ {name} = {score} is not a valid number (type: {type(score).__name__})")
                    all_valid = False
        
        return all_valid
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                print(f"  Response status: {e.response.status_code}")
                print(f"  Response body: {e.response.text[:200]}")
            except:
                pass
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        print(f"  Traceback: {traceback.format_exc()}")
        return False


def test_health_check():
    """Test that the service is running."""
    print("=" * 60)
    print("Health Check")
    print("=" * 60)
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        response.raise_for_status()
        data = response.json()
        print(f"✓ Service is healthy: {data}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"✗ Service is not responding: {e}")
        print(f"  Make sure the service is running on {API_URL}")
        return False


def diagnose_verification_issue():
    """Run a diagnostic query to see what's happening with verification."""
    print("\n" + "=" * 60)
    print("Diagnostic: Checking Verification Status")
    print("=" * 60)
    
    query = {
        "query": "What was Apple's revenue in 2023?",
        "filters": {"ticker": "AAPL", "year": "2023"},
        "top_k": 5,
        "enable_verification": True
    }
    
    try:
        print(f"\nMaking test query to {API_URL}/query...")
        response = requests.post(f"{API_URL}/query", json=query, timeout=120)
        response.raise_for_status()
        data = response.json()
        
        print("\nResponse structure:")
        print(f"  Keys in response: {list(data.keys())}")
        
        if "verification" in data:
            verification = data["verification"]
            print("\n✓ Verification object found!")
            print(f"  Verification keys: {list(verification.keys())}")
            print(f"  Verification content: {json.dumps(verification, indent=2)}")
        else:
            print("\n✗ Verification object NOT found")
            print("\n  Full response structure:")
            print(f"  {json.dumps(data, indent=2)[:1000]}...")
            
        if "answer" in data:
            print(f"\n  Answer length: {len(data['answer'])} chars")
        if "sources" in data:
            print(f"  Sources count: {len(data.get('sources', []))}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Diagnostic failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def main():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("RAG Verification System Test")
    print("=" * 60)
    print(f"\nTesting API at: {API_URL}")
    print("(Change API_URL in the script if using Docker on port 8000)\n")
    
    results = []
    
    # Health check first
    if not test_health_check():
        print("\n✗ Service is not available. Please start the service first.")
        sys.exit(1)
    
    # Run diagnostic if verification tests fail
    verification_enabled_result = test_verification_enabled()
    verification_disabled_result = test_verification_disabled()
    score_validation_result = test_verification_scores()
    
    # If verification enabled test failed, run diagnostics
    if not verification_enabled_result:
        print("\n" + "=" * 60)
        print("Running diagnostics...")
        print("=" * 60)
        diagnose_verification_issue()
    
    # Run tests
    results.append(("Health Check", True))  # Already passed
    results.append(("Verification Enabled", verification_enabled_result))
    results.append(("Verification Disabled", verification_disabled_result))
    results.append(("Score Validation", score_validation_result))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Verification system is working correctly.")
        return 0
    else:
        print("\n✗ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

