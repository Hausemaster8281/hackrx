# Test script for the HackRx API
import requests
import json

# Configuration
API_URL = "http://localhost:8000/hackrx/run"
API_TOKEN = "93f02c19721b0c40d273b9395370f249d43c2d3deb2a39498fdd8a2b9e2d3ee3"

# Test data
test_request = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?"
    ]
}

# Headers
headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Authorization": f"Bearer {API_TOKEN}"
}

def test_api():
    print("🧪 Testing HackRx API...")
    print(f"📡 URL: {API_URL}")
    print(f"❓ Questions: {len(test_request['questions'])}")
    
    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=test_request,
            timeout=60
        )
        
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"📝 Answers received: {len(result['answers'])}")
            
            for i, answer in enumerate(result['answers']):
                print(f"\n❓ Question {i+1}: {test_request['questions'][i]}")
                print(f"💡 Answer: {answer[:100]}...")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"💬 Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out")
    except requests.exceptions.ConnectionError:
        print("🔌 Connection error - is the server running?")
    except Exception as e:
        print(f"💥 Unexpected error: {str(e)}")

if __name__ == "__main__":
    test_api()
