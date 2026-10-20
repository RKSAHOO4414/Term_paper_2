import json
import sys
import re

def strip_ansi(text):
    return re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)

def safe_print(text):
    text = strip_ansi(text)
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', errors='replace').decode('ascii'))

def read_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb.get('cells', [])
    safe_print(f"\n{'='*80}")
    safe_print(f"NOTEBOOK: {path}")
    safe_print(f"Total cells: {len(cells)}")
    safe_print(f"{'='*80}\n")
    
    for i, cell in enumerate(cells):
        ct = cell.get('cell_type', 'unknown')
        source = ''.join(cell.get('source', []))
        safe_print(f"\n{'='*60}")
        safe_print(f"CELL {i} [{ct.upper()}]")
        safe_print(f"{'='*60}")
        safe_print(source)
        
        outputs = cell.get('outputs', [])
        if outputs:
            safe_print(f"\n--- OUTPUT ---")
            for o in outputs:
                if 'text' in o:
                    safe_print(''.join(o['text']))
                elif 'data' in o:
                    if 'text/plain' in o['data']:
                        safe_print(''.join(o['data']['text/plain']))
                    if 'text/html' in o['data']:
                        safe_print('[HTML output]')
                    if 'image/png' in o['data']:
                        safe_print('[Image output]')
                if 'ename' in o:
                    safe_print(f"ERROR: {o.get('ename')}: {o.get('evalue')}")
        safe_print("")

if __name__ == '__main__':
    for path in sys.argv[1:]:
        read_notebook(path)
