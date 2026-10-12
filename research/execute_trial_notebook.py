from pathlib import Path
import sys
root=Path('D:/VLM-maps');sys.path.insert(0,str(root/'.packages'))
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
n=nbformat.read(root/'notebookgyu-cpu.next-trials.ipynb',as_version=4)
km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
c=NotebookClient(n,timeout=600,resources={'metadata':{'path':str(root)}});c.km=km
c.execute();nbformat.write(n,root/'notebookgyu-cpu.next-trials.executed.ipynb')
errors=[o for cell in n.cells if cell.cell_type=='code' for o in cell.get('outputs',[]) if o.output_type=='error']
assert not errors
print('All 5 trial cells executed; no errors',flush=True)
