# step3_gpt_process.py
import json
import openai
import time
import argparse
import logging
from pathlib import Path

def setup_logging():
    """Configure logging for the processor"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("translation_processor.log"),
            logging.StreamHandler()
        ]
    )

def validate_input_files(*files):
    """Ensure all input files exist before processing"""
    for file in files:
        if not Path(file).exists():
            raise FileNotFoundError(f"Critical file missing: {file}")

def validate_output(input_file, output_file):
    """Validate that output has same number of entries as input"""
    with open(input_file, 'r', encoding='utf-8') as f:
        input_entries = [e for e in f.read().split("\n\n") if e.strip()]
    
    with open(output_file, 'r', encoding='utf-8') as f:
        output_entries = [e for e in f.read().split("\n\n") if e.strip()]
    
    if len(input_entries) != len(output_entries):
        raise ValueError(
            f"Entry count mismatch! Input: {len(input_entries)}, Output: {len(output_entries)}"
        )
    return True

def sanitize_gpt_output(text, target_lang):
    """Clean GPT's output to ensure consistent formatting"""
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("```") or line.endswith("```"):
            continue
        if line.lower().startswith(target_lang.lower()):
            line = f"{target_lang}: {line.split(':', 1)[1].strip()}"
        lines.append(line)
    return "\n".join(lines)

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

def process_with_api(input_file, output_file, api_key, args, max_retries=3, batch_size=10):
    """Process translations with batch processing"""
    validate_input_files(input_file)
    
    client = openai.OpenAI(api_key=api_key)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = [entry.strip() for entry in f.read().split("\n\n") if entry.strip()]
    

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
Decision Criteria EXAMPLE:
Input format you'll receive:
```
BLOCK_123 | tag_name
en: Log in to your account
fr: Connecter à votre compte
```
✓ Do IMPROVE (grammatical error): `fr: Se connecter à votre compte`
✓ Do not IMPROVE (already good): If translation was already `Se connecter à votre compte`

**OUTPUT FORMAT:**
Return ONLY the improved {args.target_lang} translation:
[your_translation]
DO NOT include:
   - HTML tags, unless they appear explicit in the current {args.target_lang} translation
   - Explanations or additional text
Correct output format examples:
- Input: `en: Submit` → Output: `Envoyer`
- Input: `zh: 苹果` → Output:`苹果`
- Input: `en: <p>Text</p>` → Output: `<p>Texte</p>`
Incorrect output format examples: 
- Input: `en: Text` → Output: `<p>Texte</p>` (added HTML)
- Input: `en: <p>Text</p>` → Output: `Texte` (removed HTML)
"""

    batches = [content[i:i+batch_size] for i in range(0, len(content), batch_size)]
    results = []
    
    for batch_idx, batch in enumerate(batches):
        logging.info(f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} entries)")
        
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
                batch_response = sanitize_gpt_output(batch_response, args.target_lang)
                
                # Split response while preserving entry structure
                batch_entries = []
                current_entry = []
                for line in batch_response.split("\n"):
                    if line.strip() == "" and current_entry:
                        batch_entries.append("\n".join(current_entry))
                        current_entry = []
                    elif line.strip():
                        current_entry.append(line)
                if current_entry:
                    batch_entries.append("\n".join(current_entry))
                
                if len(batch_entries) != len(batch):
                    raise ValueError(f"Expected {len(batch)} entries, got {len(batch_entries)}")
                
                results.extend(batch_entries)
                break
                
            except Exception as e:
                logging.warning(f"Batch {batch_idx + 1} attempt {attempt + 1} failed: {str(e)[:100]}")
                if attempt == max_retries - 1:
                    logging.info(f"Processing batch {batch_idx + 1} individually as fallback")
                    for entry in batch:
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
                            improved_trans = response.choices[0].message.content.strip()
                            improved_trans = sanitize_gpt_output(improved_trans, args.target_lang)
                            results.append(f"{entry}\n{args.target_lang}: {improved_trans}")
                        except Exception as individual_error:
                            logging.error(f"Error processing entry: {str(individual_error)[:50]}")
                            results.append(f"{entry}\n# ERROR: {str(individual_error)[:50]}")
                        time.sleep(1)
                else:
                    time.sleep(2 ** attempt)
        
        time.sleep(1)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(results))

def parse_gpt_output(gpt_output_file, target_lang):
    """Parse GPT output into a translations dictionary"""
    validate_input_files(gpt_output_file)
    
    translations = {}
    with open(gpt_output_file, 'r', encoding='utf-8') as f:
        entries = [entry.strip() for entry in f.read().split("\n\n") if entry.strip()]
    
    for entry in entries:
        if not entry.strip() or entry.startswith("#"):
            continue
        
        lines = [line.strip() for line in entry.split("\n") if line.strip()]
        if len(lines) < 2:
            continue
        
        if " | " in lines[0]:
            block_id = lines[0].split(" | ")[0].strip()
            target_lines = [line for line in lines if line.startswith(f"{target_lang}:")]
            if target_lines:
                final_trans = target_lines[-1].split(":", 1)[1].strip()
                translations[block_id] = final_trans
    
    return translations

if __name__ == "__main__":
    setup_logging()
    logging.info("Starting translation processing")
    
    parser = argparse.ArgumentParser(description="GPT Translation Processor")
    parser.add_argument("--context", required=True, help="translatable_flat_sentences.json")
    parser.add_argument("--translated", required=True, help="segments_only.json")
    parser.add_argument("--output", default="gpt_processed.txt")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--primary-lang", required=True)
    parser.add_argument("--secondary-lang")
    parser.add_argument("--target-lang", required=True)
    parser.add_argument("--batch-size", type=int, default=10, help="Number of entries per batch")
    
    args = parser.parse_args()
    
    try:
        validate_input_files(args.context, args.translated)
        
        intermediate_file = "gpt_input.txt"
        build_gpt_friendly_input(
            args.context,
            args.translated,
            intermediate_file,
            args.target_lang,
            args.primary_lang
        )
        
        process_with_api(
            intermediate_file,
            args.output,
            args.api_key,
            args,
            batch_size=args.batch_size
        )
        
        validate_output(intermediate_file, args.output)
        
        final_translations = parse_gpt_output(args.output, args.target_lang)
        
        with open("openai_translations.json", "w", encoding="utf-8") as f:
            json.dump(final_translations, f, indent=2, ensure_ascii=False)
        
        logging.info("✅ Successfully saved openai_translations.json")
    
    except Exception as e:
        logging.error(f"❌ Processing failed: {str(e)}")
        raise
