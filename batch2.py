# step3_gpt_process.py
import json
import openai
import time
import argparse
from pathlib import Path

def validate_input_files(*files):
    """Ensure all input files exist before processing"""
    for file in files:
        if not Path(file).exists():
            raise FileNotFoundError(f"Critical file missing: {file}")

def build_gpt_friendly_input(context_file, translated_file, output_file, target_lang, primary_lang):
    """Generate GPT-ready input with language context"""
    validate_input_files(context_file, translated_file)
    
    with open(context_file, 'r', encoding='utf-8') as f:
        context_data = json.load(f)
    
    with open(translated_file, 'r', encoding='utf-8') as f:
        translated_map = json.load(f)
    
    lines = []
    for category in ['1_word', '2_words', '3_words', '4_or_more_words']:
        for entry in context_data[category]:
            tag = entry['tag']
            block_ids = []
            for key in entry.keys():
                if key != 'tag':
                    # Split merged block IDs (e.g., "BLOCK_1=BLOCK_2")
                    block_ids.extend(key.split('='))
            
            for block_id in block_ids:
                source_text = entry.get(block_id, "")
                translated_text = translated_map.get(block_id, "")
                
                lines.append(f"{block_id} | {tag}")
                lines.append(f"{primary_lang}: {source_text}")
                lines.append(f"{target_lang}: {translated_text}")
                lines.append("")  # Empty line separator
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

def process_with_api_direct_json(input_file, api_key, args, max_retries=3, batch_size=10):
    """Process translations with batch processing and write directly to JSON"""
    validate_input_files(input_file)
    
    client = openai.OpenAI(api_key=api_key)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read().split("\n\n")
    
    system_prompt = f"""You are a professional translator. You will receive entries with:
1. Original text in {args.primary_lang}{f" or {args.secondary_lang}" if args.secondary_lang else ""}
2. Current {args.target_lang} translation

Compare the original with the current translation to determine if improvement is needed.

**TRANSLATION SCOPE AND LANGUAGE IDENTIFICATION:**
-Only translate text if the original text is in:
- **{args.primary_lang}**: Translate to {args.target_lang}
{f"- **{args.secondary_lang}**: Translate to {args.target_lang}" if args.secondary_lang else ""}
- **For Any other language**: Return the original text unchanged.

**EVALUATION PROCESS:**
1. Compare the original text with the current {args.target_lang} translation
2. Identify if the current translation has issues:
   - **Accuracy**: Wrong meaning, missing information, mistranslations
   - **Naturalness**: Awkward phrasing, overly literal translation
   - **Grammar**: Incorrect verb forms, word order, agreement errors
   - **Terminology**: Inconsistent or inappropriate word choices
   - **Context**: Doesn't fit UI/web context appropriately

**DECISION CRITERIA:**
- **Do IMPROVE**: If current translation has any of the above issues
- **Do not IMPROVE**: If current translation is accurate, natural, and appropriate

**CRITICAL OUTPUT FORMAT:**
You MUST return a valid JSON object with BLOCK_ID as keys and improved translations as values:
{{
  "BLOCK_X": "improved_translation_1",
  "BLOCK_Y": "improved_translation_2"
}}

DO NOT include:
- HTML tags, unless they appear explicit in the current {args.target_lang} translation
- Explanations or additional text
- Any text outside the JSON object

**EXAMPLES:**
Input:
```
BLOCK_123 | tag_name
en: Log in
fr: Connecter
```

Output:
```json
{{
  "BLOCK_123": "Se connecter"
}}
```"""

    # Group entries into batches
    entries = [entry.strip() for entry in content if entry.strip()]
    batches = [entries[i:i+batch_size] for i in range(0, len(entries), batch_size)]
    
    final_translations = {}
    
    for batch_idx, batch in enumerate(batches):
        print(f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} entries)")
        
        batch_input = "\n\n".join(batch)
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": batch_input}
                    ],
                    temperature=0.2,
                    max_tokens=4000
                )
                
                batch_response = response.choices[0].message.content.strip()
                
                # Try to parse JSON response
                try:
                    # Clean response (remove code blocks if present)
                    if batch_response.startswith('```json'):
                        batch_response = batch_response.split('```json')[1].split('```')[0].strip()
                    elif batch_response.startswith('```'):
                        batch_response = batch_response.split('```')[1].split('```')[0].strip()
                    
                    batch_translations = json.loads(batch_response)
                    final_translations.update(batch_translations)
                    print(f"✅ Batch {batch_idx + 1} processed successfully ({len(batch_translations)} translations)")
                    break
                    
                except json.JSONDecodeError as json_error:
                    print(f"⚠️ Batch {batch_idx + 1} JSON parsing failed: {str(json_error)[:100]}")
                    if attempt == max_retries - 1:
                        # Fallback: process individually
                        print(f"🔄 Processing batch {batch_idx + 1} individually as fallback")
                        for entry in batch:
                            individual_translation = process_individual_entry(client, system_prompt, entry, args)
                            if individual_translation:
                                final_translations.update(individual_translation)
                        break
                    else:
                        time.sleep(2 ** attempt)
                        continue
                        
            except Exception as e:
                print(f"❌ Batch {batch_idx + 1} attempt {attempt + 1} failed: {str(e)[:100]}")
                if attempt == max_retries - 1:
                    # Fallback: process individually
                    print(f"🔄 Processing batch {batch_idx + 1} individually as fallback")
                    for entry in batch:
                        individual_translation = process_individual_entry(client, system_prompt, entry, args)
                        if individual_translation:
                            final_translations.update(individual_translation)
                else:
                    time.sleep(2 ** attempt)
        
        time.sleep(1)  # Rate limit buffer between batches
    
    return final_translations

def process_individual_entry(client, system_prompt, entry, args):
    """Process a single entry and return translation dict"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": entry}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        
        individual_response = response.choices[0].message.content.strip()
        
        # Try to parse as JSON first
        try:
            if individual_response.startswith('```json'):
                individual_response = individual_response.split('```json')[1].split('```')[0].strip()
            elif individual_response.startswith('```'):
                individual_response = individual_response.split('```')[1].split('```')[0].strip()
            
            return json.loads(individual_response)
        except json.JSONDecodeError:
            # Fallback: extract BLOCK_ID and use response as translation
            lines = entry.split('\n')
            if lines and '|' in lines[0]:
                block_id = lines[0].split('|')[0].strip()
                return {block_id: individual_response}
        
    except Exception as e:
        print(f"❌ Individual entry failed: {str(e)[:50]}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT Translation Processor")
    parser.add_argument("--context", required=True, help="translatable_flat_sentences.json")
    parser.add_argument("--translated", required=True, help="segments_only.json")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--primary-lang", required=True)
    parser.add_argument("--secondary-lang")
    parser.add_argument("--target-lang", required=True)
    parser.add_argument("--batch-size", type=int, default=5, help="Number of entries per batch")
    
    args = parser.parse_args()
    
    # Validate input files first
    validate_input_files(args.context, args.translated)
    
    # Generate GPT-ready input
    intermediate_file = "gpt_input.txt"
    build_gpt_friendly_input(
        args.context,
        args.translated,
        intermediate_file,
        args.target_lang,
        args.primary_lang
    )
    
    # Process with API and get final translations directly
    final_translations = process_with_api_direct_json(
        intermediate_file,
        args.api_key,
        args,
        batch_size=args.batch_size
    )
    
    # Save final translations
    with open("openai_translations.json", "w", encoding="utf-8") as f:
        json.dump(final_translations, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(final_translations)} translations to openai_translations.json")
