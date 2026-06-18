import re

with open("main_qwen.py", "r", encoding="utf-8") as f:
    text = f.read()

# 1. Imports
text = text.replace(
    'import google.generativeai as genai\nfrom dotenv import load_dotenv\nload_dotenv()',
    'from openai import OpenAI\nimport base64\nfrom dotenv import load_dotenv\nload_dotenv()'
)

# 2. Config
text = text.replace(
    '# Configure Gemini\napi_key = os.environ.get("GEMINI_API_KEY")\nif not api_key:\n    raise EnvironmentError("GEMINI_API_KEY is not set. Add it to your .env file.")\ngenai.configure(api_key=api_key)\nmodel = genai.GenerativeModel("gemini-2.5-flash")',
    '''# Configure Local LLM (Qwen2.5-VL)
client = OpenAI(base_url="http://localhost:11434/v1", api_key="local-llm")
VISION_MODEL = "qwen2.5-vl-3b-instruct"

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')'''
)

# 3. Batch variables to LLM_FRAME_INTERVAL
text = re.sub(
    r'# GEMINI DENSE ANALYSIS\s*\(every 7 seconds\)\s*# =+.*?\nGEMINI_FRAME_INTERVAL = \d+.*?\nGEMINI_BATCH_SIZE = \d+.*?\nGEMINI_BATCH_WORKERS = \d+.*?\n\nGEMINI_PER_FRAME_PROMPT',
    '# LOCAL LLM DENSE ANALYSIS (every 4 seconds)\n# =========================\nLLM_FRAME_INTERVAL = 4\n\nLLM_PER_FRAME_PROMPT',
    text,
    flags=re.DOTALL
)

# 4. Remove chunked, build_gemini_batch_prompt, parse_gemini_batch_response, analyze_frame_batch_with_gemini
# Finding the block from def chunked to the start of analyze_with_gemini_dense
start_str = "def chunked(items, size):"
end_str = "def analyze_with_gemini_dense(parameters, frame_analyses):"
start_idx = text.find(start_str)
end_idx = text.find(end_str)

if start_idx != -1 and end_idx != -1:
    # Build single frame analysis function
    new_func = """
def analyze_single_frame_with_llm(frame_idx, frame_analyses):
    \"\"\"Analyze a single frame using the local OpenAI endpoint.\"\"\"
    image_path = frame_analyses[frame_idx]["frame_path"]
    base64_image = encode_image_to_base64(image_path)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": LLM_PER_FRAME_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=messages,
            max_tokens=1000,
            temperature=0.2
        )
        response_text = response.choices[0].message.content.strip()
        return parse_gemini_per_frame(response_text)
    except Exception as e:
        print(f"Error during LLM call for frame {frame_idx}: {e}")
        return {}

"""
    text = text[:start_idx] + new_func + text[end_idx:]

# 5. Overhaul analyze_with_gemini_dense
# It starts at `def analyze_with_gemini_dense`
text = text.replace('def analyze_with_gemini_dense(', 'def analyze_with_llm_dense(')
text = text.replace('GEMINI_FRAME_INTERVAL', 'LLM_FRAME_INTERVAL')

# We need to replace the ThreadPoolExecutor batching with a sequential loop
executor_start = "batches = list(chunked(gemini_indices, GEMINI_BATCH_SIZE))"
executor_end = "    for idx in gemini_indices:"
start_idx = text.find(executor_start)
end_idx = text.find(executor_end)

if start_idx != -1 and end_idx != -1:
    new_loop = """
    # Process sequentially (no batching/multithreading for local LLM)
    for i, idx in enumerate(gemini_indices, start=1):
        timestamp = frame_analyses[idx]["timestamp"]
        print(f"  Analyzing frame {idx} ({timestamp}s) with Local LLM ({i}/{len(gemini_indices)})...")
        parsed = analyze_single_frame_with_llm(idx, frame_analyses)
        
        gemini_results[idx] = {
            "idx": idx,
            "timestamp": timestamp,
            "parsed": parsed,
            "error": None
        }

"""
    text = text[:start_idx] + new_loop + text[end_idx:]


# Replace Gemini references outside structure
text = text.replace('parse_gemini_per_frame', 'parse_llm_per_frame')
text = text.replace('GEMINI_PER_FRAME_PROMPT', 'LLM_PER_FRAME_PROMPT')
text = text.replace('Gemini votes', 'LLM votes')
text = text.replace('Gemini evidence', 'LLM evidence')
text = text.replace('Gemini requests', 'LLM requests')
text = text.replace('Gemini results', 'llm_results')
text = text.replace('gemini_summary', 'llm_summary')
text = text.replace('gemini_recs', 'llm_recs')
text = text.replace('gemini_detail', 'llm_detail')
text = text.replace('gemini_indices', 'llm_indices')
text = text.replace('gemini_votes', 'llm_votes')
text = text.replace('gemini_reasons', 'llm_reasons')
text = text.replace('gemini_evidence', 'llm_evidence')
text = text.replace('gemini_total', 'llm_total')
text = text.replace('gemini_alertness', 'llm_alertness')
text = text.replace('gemini_status', 'llm_status')
text = text.replace('analyze_with_gemini_dense', 'analyze_with_llm_dense')
text = text.replace('analyze_with_llm_dense', 'analyze_with_llm_dense')  # already done but just in case
text = text.replace('Gemini analysis skipped', 'Local LLM analysis skipped')
text = text.replace('Gemini dense analysis complete', 'Local LLM dense analysis complete')
text = text.replace('Gemini AI', 'Local LLM')
text = text.replace('Gemini response parsing', 'LLM response parsing')


with open("main_qwen.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Successfully generated main_qwen.py!")
