from pathlib import Path
import json
p=Path('notebookgyu-cpu.room26.executed.ipynb');n=json.loads(p.read_text(encoding='utf-8'));cells=[c for c in n['cells'] if c['cell_type']=='code'];errors=[o for c in cells for o in c.get('outputs',[]) if o['output_type']=='error']
print('Executed cells:',[c['execution_count'] for c in cells],'; errors:',len(errors));assert len(cells)==5 and not errors and all(c['execution_count'] for c in cells)
r=Path('research/ROOM_UPGRADE_REPORT.md');s=r.read_text(encoding='utf-8').replace('All **12 regression tests passed**','All **five notebook code cells executed without errors**. All **12 regression tests passed**');r.write_text(s,encoding='utf-8')
