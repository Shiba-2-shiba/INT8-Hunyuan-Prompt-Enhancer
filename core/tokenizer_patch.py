import os
import shutil

def patch_tokenizer(model_path):
    """
    Patches `tokenization_hy.py` to fix the initialization order bug in Hunyuan's remote code.
    Critical for avoiding runtime errors.
    """
    tokenizer_file = os.path.join(model_path, "tokenization_hy.py")
    if not os.path.exists(tokenizer_file):
        return
    
    with open(tokenizer_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if patch is needed (super().__init__ called before mergeable_ranks init)
    if "self.mergeable_ranks = _load_tiktoken_bpe(vocab_file)" in content and \
       content.find("self.mergeable_ranks") < content.find("super().__init__(**kwargs)"):
        return # Already correct

    # Backup
    if not os.path.exists(tokenizer_file + ".bak"):
        try:
            shutil.copy2(tokenizer_file, tokenizer_file + ".bak")
        except Exception:
            pass

    # Simple line swap patch strategy
    lines = content.splitlines()
    super_idx = -1
    ranks_idx = -1
    
    for i, line in enumerate(lines):
        if "super().__init__(**kwargs)" in line:
            super_idx = i
        if "self.mergeable_ranks = _load_tiktoken_bpe(vocab_file)" in line:
            ranks_idx = i
            break 
    
    if super_idx != -1 and ranks_idx != -1 and super_idx < ranks_idx:
        print(f"Patching tokenizer at {tokenizer_file}")
        super_line = lines[super_idx]
        lines.pop(super_idx)
        # Re-calculate index since we popped
        ranks_idx -= 1
        lines.insert(ranks_idx + 1, super_line)
        
        with open(tokenizer_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
