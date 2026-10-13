import os

MODEL_DIR = "models/parakeet-tdt-0.6b-v3-int8"
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.txt")
TOKENS_PATH = os.path.join(MODEL_DIR, "tokens.txt")

def fix_tokens():
    if not os.path.exists(VOCAB_PATH):
        print(f"Error: {VOCAB_PATH} not found")
        return

    print(f"Reading {VOCAB_PATH}...")
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    print(f"Original line count: {len(lines)}")
    
    # Check for <blk>
    blk_index = -1
    clean_lines = []
    
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) >= 1:
            token = parts[0]
            if token == "<blk>":
                blk_index = i
                print(f"Found <blk> at index {i}")
            else:
                clean_lines.append(line)
        else:
            # Empty line?
            clean_lines.append(line)

    # We need total 8193 tokens (indices 0 to 8192)
    # The last one (8192) MUST be <blk>
    # So we need 8192 tokens before it (indices 0 to 8191)
    
    target_vocab_size = 8193
    target_pre_blk_count = target_vocab_size - 1 # 8192
    
    current_count = len(clean_lines)
    missing_count = target_pre_blk_count - current_count
    
    print(f"Clean count (without <blk>): {current_count}")
    print(f"Target count before <blk>: {target_pre_blk_count}")
    print(f"Missing tokens: {missing_count}")
    
    if missing_count > 0:
        print(f"Adding {missing_count} placeholder tokens...")
        for i in range(missing_count):
            clean_lines.append(f"<unused_{i}> {current_count + i}")
            
    # Add <blk> at the end
    # Sherpa-ONNX expects <blk> to be at the very end
    # IMPORTANT: Use strict "<blk> {index}" format and ensure it's the LAST LINE
    # Some parsers might be sensitive to whitespace or token content
    
    clean_lines.append(f"<blk> {target_vocab_size - 1}")
    
    # Validation
    if len(clean_lines) != target_vocab_size:
        print(f"ERROR: Generated {len(clean_lines)} tokens, expected {target_vocab_size}")
    
    # Double check the last item
    last_item = clean_lines[-1]
    if not last_item.startswith("<blk>"):
         print(f"ERROR: Last token is '{last_item}', expected it to start with <blk>")
    
    print(f"Final token count: {len(clean_lines)}")
    print(f"Last token: {clean_lines[-1]}")
    
    print(f"Writing to {TOKENS_PATH}...")
    with open(TOKENS_PATH, "w", encoding="utf-8") as f:
        # Join with newlines, but NO trailing newline to avoid empty last line reading
        f.write("\n".join(clean_lines))
        
    print("Done!")

if __name__ == "__main__":
    fix_tokens()
