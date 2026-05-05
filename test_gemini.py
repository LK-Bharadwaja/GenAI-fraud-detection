import sys, os, traceback
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

key = os.getenv('GOOGLE_API_KEY', '')
print(f"API key loaded: {bool(key)} | first 6: {key[:6]} | length: {len(key)}")

print("\n--- Test 1: Direct google-genai SDK ---")
try:
    from google import genai
    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents='Say hello in one sentence.'
    )
    print("SUCCESS:", response.text)
except Exception as e:
    print("FAILED:")
    traceback.print_exc()

print("\n--- Test 2: LangChain wrapper ---")
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.4,
        max_output_tokens=300,
        google_api_key=key,
    )
    response = llm.invoke("Say hello in one sentence.")
    print("SUCCESS:", response.content)
except Exception as e:
    print("FAILED:")
    traceback.print_exc()

print("\n--- Test 3: Full LangChainExplanationEngine ---")
try:
    from genai_explanations.langchain_explainer import LangChainExplanationEngine
    import pandas as pd
    engine = LangChainExplanationEngine()
    print(f"llm_source: {engine.llm_source}")
    print(f"llm object is None: {engine.llm is None}")
    test_df = pd.DataFrame([{
        'anomaly_id': 'A0001', 'entity_type': 'transaction', 'entity_id': 'TX001',
        'anomaly_type': 'amount_zscore', 'severity': 'HIGH',
        'reason': 'Transaction amount deviates 6.2 std from mean'
    }])
    result = engine.generate_explanations(test_df)
    print(f"llm_source in output: {result.iloc[0]['llm_source']}")
    print(f"Explanation:\n{result.iloc[0]['llm_explanation']}")
except Exception as e:
    print("FAILED:")
    traceback.print_exc()
