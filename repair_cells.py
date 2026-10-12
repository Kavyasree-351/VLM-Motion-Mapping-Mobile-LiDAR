import os,json
from pathlib import Path
import nbformat
from nbclient import NotebookClient
root=Path.cwd(); os.environ['PYTHONPATH']=str(root/'.packages'); os.environ['MPLCONFIGDIR']=str(root/'.matplotlib')
local=nbformat.read('notebookgyu-cpu.local.ipynb',as_version=4)
repair=nbformat.v4.new_notebook(cells=[local.cells[i] for i in [28,47]])
NotebookClient(repair,timeout=None,kernel_name='python3',resources={'metadata':{'path':str(root)}}).execute()
nbformat.write(repair,'repaired_cells.ipynb')
print('Both repaired cells passed',flush=True)
