import os
import json
import deepl
import argparse
import re
from pathlib import Path


def create_efficient_translatable_map(
    json_data, 
    translator, 
    target_lang="FR", 
    primary_lang=None, 
    secondary_lang=None, 
    memory_file=None
):
    """
    Creates a translation map with language validation.
    Only translates text detected as primary_lang or secondary_lang.
    """
    # Load translation memory
    translation_memory = {}
    if memory_file and os.path.exists(memory_file):
        try:
            with open(memory_file, "r", encoding="utf-8") as f:
                translation_memory = json.load(f)
            print(f"Loaded {len(translation_memory)} cached translations")
        except json.JSONDecodeError:
            print(f"Warning: Corrupted translation memory file {memory_file}")

    # Prepare translation data structures
    translatable_map = {}
    texts_to_translate = []
    token_indices = []
    original_texts = {}

    # Process all blocks and segments
    for block_id, block_data in json_data.items():
        if "text" in block_data:
            text = block_data["text"]
            token = block_id
            if text in translation_memory:
                translatable_map[token] = translation_memory[text]
                print(f"Using cached: {token}")
            else:
                texts_to_translate.append(text)
                token_indices.append(token)
                original_texts[token] = text

        if "segments" in block_data:
            for segment_id, segment_text in block_data["segments"].items():
                token = f"{block_id}_{segment_id}"
                if segment_text in translation_memory:
                    translatable_map[token] = translation_memory[segment_text]
                    print(f"Using cached segment: {token}")
                else:
                    texts_to_translate.append(segment_text)
                    token_indices.append(token)
                    original_texts[token] = segment_text

    
    def clean_text(text):
        """Clean text for language detection only"""
        text = re.sub(r'^(.*?):\s*', '', text)  # Remove prefixes
        text = re.sub(r'[^\w\sÀ-ÿ=+-]', ' ', text)
        text = re.sub(r'[^\w\sà-üÀ-Ü]', ' ', text)  # Clean special chars
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with one
        text = re.sub(r'^\W+|\W+$', '', text)  # Trim edges
        return text.strip()[:500]  # Limit for detection

    # Language-aware batch translation
    if texts_to_translate:
        print(f"Processing {len(texts_to_translate)} segments with language validation...")
        
        batch_size = 330
        for batch_idx in range(0, len(texts_to_translate), batch_size):
            batch = texts_to_translate[batch_idx:batch_idx+batch_size]
            translated_batch = []
            
            try:
                # Phase 1: Language detection with cleaned text
                detection_texts = [clean_text(text) for text in batch]
                translation_texts = batch  # Keep original texts for translation
                
                detection_results = translator.translate_text(
                    detection_texts,
                    target_lang=target_lang,
                    preserve_formatting=True
                )

                # Phase 2: Translation with original texts
                for idx, detection in enumerate(detection_results):
                    detected_lang = detection.detected_source_lang.lower()
                    allowed_langs = {lang.lower() for lang in [primary_lang, secondary_lang] if lang}
                    original_text = translation_texts[idx]

                    # Short-text bypass
                    if len(original_text.strip()) < 15 and secondary_lang:
                        try:
                            result = translator.translate_text(original_text, target_lang=target_lang)
                            translated_batch.append(result.text)
                            continue
                        except Exception as e:
                            translated_batch.append(original_text)
                            continue

                    if allowed_langs and detected_lang in allowed_langs:
                        result = translator.translate_text(original_text, target_lang=target_lang)
                        translated_batch.append(result.text)
                    else:
                        translated_batch.append(original_text)

            except Exception as e:
                print(f"Translation skipped for batch (error: {str(e)[:50]}...)")
                translated_batch.extend(batch)
            
            # Store results
            for j in range(len(batch)):
                global_index = batch_idx + j
                token = token_indices[global_index]
                original_text = original_texts[token]
                final_text = translated_batch[j]
                
                translatable_map[token] = final_text
                translation_memory[original_text] = final_text
            
            print(f"Completed batch {batch_idx//batch_size + 1}/{(len(texts_to_translate) + batch_size - 1)//batch_size}")

    # Update translation memory
    if memory_file and translation_memory:
        os.makedirs(os.path.dirname(memory_file), exist_ok=True)
        with open(memory_file, "w", encoding="utf-8") as f:
            json.dump(translation_memory, f, ensure_ascii=False, indent=2)
        print(f"Updated translation memory with {len(translation_memory)} entries")

    return translatable_map




def translate_json_file(
    input_file, 
    output_file, 
    target_lang="FR", 
    primary_lang=None, 
    secondary_lang=None, 
    memory_dir="translation_memory",
    segment_file=None
):
    """Main translation function with language validation"""
    # Auth check
    auth_key = os.getenv("DEEPL_AUTH_KEY")
    if not auth_key:
        raise ValueError("DEEPL_AUTH_KEY environment variable not set")

    # Initialize translator
    translator = deepl.Translator(auth_key)
    
    # Create memory directory
    os.makedirs(memory_dir, exist_ok=True)
    memory_file = os.path.join(memory_dir, f"translation_memory_{target_lang.lower()}.json")

    # Load input data
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            json_data = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load {input_file}: {e}")

    # Create translation map
    translatable_map = create_efficient_translatable_map(
        json_data=json_data,
        translator=translator,
        target_lang=target_lang,
        primary_lang=primary_lang,
        secondary_lang=secondary_lang,
        memory_file=memory_file
    )

    # Rebuild structure with translations
    translated_data = {}
    for block_id, block_data in json_data.items():
        translated_block = block_data.copy()
        
        if "text" in block_data:
            translated_block["text"] = translatable_map.get(block_id, block_data["text"])
        
        if "segments" in block_data:
            translated_segments = {
                seg_id: translatable_map.get(f"{block_id}_{seg_id}", seg_text)
                for seg_id, seg_text in block_data["segments"].items()
            }
            translated_block["segments"] = translated_segments
        
        translated_data[block_id] = translated_block

    # Save output
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(translated_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Translation completed: {output_file}")
    
    # Export segments if requested
    if segment_file:
        segment_translations = {}
        for block_id, block_data in translated_data.items():
            if "segments" in block_data:
                for seg_id, seg_text in block_data["segments"].items():
                    segment_translations[seg_id] = seg_text

        with open(segment_file, "w", encoding="utf-8") as f:
            json.dump(segment_translations, f, indent=2, ensure_ascii=False)
        print(f"✅ Segment-only translations exported: {segment_file}")

    return translated_data

def apply_translations(original_file, translations_file, output_file):
    """Applies translations to original JSON structure"""
    with open(original_file, "r", encoding="utf-8") as f:
        original_data = json.load(f)
    
    with open(translations_file, "r", encoding="utf-8") as f:
        translations = json.load(f)
    
    translated_data = {}
    for block_id, block_data in original_data.items():
        translated_block = block_data.copy()
        
        if "text" in block_data:
            translated_block["text"] = translations.get(block_id, block_data["text"])
        
        if "segments" in block_data:
            translated_block["segments"] = {
                seg_id: translations.get(f"{block_id}_{seg_id}", seg_text)
                for seg_id, seg_text in block_data["segments"].items()
            }
        
        translated_data[block_id] = translated_block
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(translated_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Applied translations to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Translate JSON content with language validation"
    )
    parser.add_argument("--input", "-i", default="translatable_flat.json", 
                       help="Input JSON file")
    parser.add_argument("--output", "-o", default="translations.json",
                       help="Output JSON file")
    parser.add_argument("--lang", "-l", required=True,
                       help="Target language code (e.g., FR, ES)")
    parser.add_argument("--primary-lang", 
                       help="Primary source language code (from step1)")
    parser.add_argument("--secondary-lang",
                       help="Secondary source language code (from step1)")
    parser.add_argument("--memory", "-m", default="translation_memory",
                       help="Translation memory directory")
    parser.add_argument("--apply", "-a", action="store_true",
                       help="Apply translations to original structure")
    parser.add_argument("--segments", "-s", 
                       help="Output file for segment-only translations")

    args = parser.parse_args()

    try:
        translations = translate_json_file(
            input_file=args.input,
            output_file=args.output,
            target_lang=args.lang,
            primary_lang=args.primary_lang,
            secondary_lang=args.secondary_lang,
            memory_dir=args.memory,
            segment_file=args.segments
        )

        if args.apply:
            apply_translations(
                args.input,
                args.output,
                f"translated_{args.input}"
            )

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
