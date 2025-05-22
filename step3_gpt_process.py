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

def process_with_api(input_file, output_file, api_key, args, max_retries=3):
    """Process translations with dynamic language validation"""
    validate_input_files(input_file)
    
    client = openai.OpenAI(api_key=api_key)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read().split("\n\n")
    
    system_prompt = f"""Follow these rules when improving {args.target_lang} translations:

1. **Translation Scope**:
   - Translate text ONLY if it's in:
     * {args.primary_lang} (primary language)
     {f"* {args.secondary_lang} (secondary language)" if args.secondary_lang else ""}
   - Preserve all other languages exactly as-is (e.g., Chinese → keep Chinese)

2. **Output Format**:
   Return ONLY the improved {args.target_lang} translation in this format:
   {args.target_lang}: [translation]
   
   DO NOT include:
   - HTML tags, unless they appear in {{translated_text}}
   - Explanations or additional text

Correct examples:
- Input: `en: Submit` → Output: `Envoyer`
- Input: `zh: 苹果` → Output:`苹果`
- Input: `en: <p>Text</p>` → Output: `<p>Texte</p>`
Incorrect examples: 
- Input: `en: Text` → Output: `<p>Texte</p>`
"""
    results = []
    for entry in content:
        if not entry.strip(): continue
        
        # Extract the existing translated line (3rd line)
        lines = entry.strip().split('\n')
        existing_translation = lines[2] if len(lines) >= 3 else ""
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": entry + f"\nCurrent translation: {existing_translation}"}
                    ],
                    temperature=0.2,
                    max_tokens=1000
                )
                
                improved_trans = response.choices[0].message.content.strip()
                
                # Minimal validation
                if not improved_trans.startswith(f"{args.target_lang}:"):
                    improved_trans = f"{args.target_lang}: {improved_trans.split(':')[-1].strip()}"
                
                results.append(f"{entry.strip()}\n{improved_trans}\n")
                break
                
            except Exception as e:
                if attempt == max_retries - 1:
                    results.append(f"{entry.strip()}\n# ERROR: {str(e)[:50]}\n")
                time.sleep(2 ** attempt)
        
        time.sleep(1)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(results))


def parse_gpt_output(gpt_output_file, target_lang):
    """Parse GPT output into a translations dictionary"""
    validate_input_files(gpt_output_file)
    
    translations = {}
    with open(gpt_output_file, 'r', encoding='utf-8') as f:
        entries = f.read().split("\n\n")
    
    for entry in entries:
        if not entry.strip():
            continue
        
        lines = entry.strip().split("\n")
        if len(lines) < 3:
            continue
        
        block_line = lines[0]
        improved_trans = next(
            (line for line in lines if line.startswith(f"{target_lang}:")),
            None
        )
        
        block_id = block_line.split(" | ")[0].strip()
        
        if improved_trans:
            final_trans = improved_trans.split(":", 1)[1].strip()
            translations[block_id] = final_trans
    
    return translations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT Translation Processor")
    parser.add_argument("--context", required=True, help="translatable_flat_sentences.json")
    parser.add_argument("--translated", required=True, help="segments_only.json")
    parser.add_argument("--output", default="gpt_processed.txt")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--primary-lang", required=True)
    parser.add_argument("--secondary-lang")
    parser.add_argument("--target-lang", required=True)
    
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
    
    # Process with API
    process_with_api(
        intermediate_file,
        args.output,
        args.api_key,
        args
    )
    
    # Generate final translations
    final_translations = parse_gpt_output(args.output, args.target_lang)
    
    with open("openai_translations.json", "w", encoding="utf-8") as f:
        json.dump(final_translations, f, indent=2, ensure_ascii=False)
    print("✅ Saved openai_translations.json")
