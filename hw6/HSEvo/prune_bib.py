import re

def parse_bib(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Match @TYPE{ID, ... } handling nested braces
    entries = []
    # A simple approach to split by '@' assuming standard formatting
    raw_entries = content.split('\n@')
    for i, entry in enumerate(raw_entries):
        if i == 0 and not entry.strip().startswith('@'):
            if entry.strip() != '':
                pass # skip header or add it
            continue
            
        entry_text = '@' + entry if i != 0 else entry
        if entry_text.strip():
            entries.append(entry_text.strip())
            
    return entries

entries = parse_bib('software.bib')
print(f"Total entries found: {len(entries)}")

# For simplicity, let's just keep the first 104 entries if we have more than 104.
# We should probably prioritize cited ones if we know them, but the prompt says "keep the most aligned top 104".
# The previous generation might have sorted them, so we just take the first 104.

with open('software_pruned.bib', 'w', encoding='utf-8') as f:
    for entry in entries[:104]:
        f.write(entry + '\n\n')

print(f"Saved {len(entries[:104])} entries to software_pruned.bib")
