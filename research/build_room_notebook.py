from pathlib import Path
import json,sys
root=Path('D:/VLM-maps')
sys.path.insert(0,str(root/'.packages'))
import nbformat
p=root/'outputs/room_refined/detector_protocol.json'
d=json.loads(p.read_text());d['models']=['YOLO11n','YOLO26m-seg','YOLOE-26l-seg','YOLOE-26l-visual'];p.write_text(json.dumps(d,indent=2))
n=nbformat.v4.new_notebook()
n.cells=[nbformat.v4.new_markdown_cell('''# iPhone LiDAR room mapping with YOLOE-26
This notebook uses the calibrated research geometry and the upgraded scene-conditioned segmentation detector. Original notebooks remain baseline artifacts. The visual reference frames are disjoint from the qualitative audit frames; this is same-scene adaptation, not a zero-shot benchmark.

Scale is inferred from exported depth and ARKit poses. Solid walls are observed segments; dashed room closure is inferred. Object surfaces remain partial. See `research/ROOM_UPGRADE_REPORT.md` for validation and limits.'''),nbformat.v4.new_code_cell('''from pathlib import Path
import sys, os, json
ROOT = Path.cwd()
if not (ROOT / 'research').exists():
    raise RuntimeError('Start this notebook in D:/VLM-maps')
sys.path.insert(0, str(ROOT / '.packages'))
sys.path.insert(0, str(ROOT / 'research'))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['MPLCONFIGDIR'] = str(ROOT / '.matplotlib')
import room_upgrade, room_views
from IPython.display import display, Image, HTML
print('Runtime:', sys.executable)
print('Chosen detector: YOLOE-26l with room visual exemplars')'''),nbformat.v4.new_code_cell('''# Uses existing inference caches. To recompute, preserve and move the relevant
# outputs/room_refined cache files before running; model weights are local.
room_views.main()
manifest = json.loads((room_upgrade.OUT / 'room_view_manifest.json').read_text())
mesh = json.loads((room_upgrade.OUT / 'mesh_metrics.json').read_text())
display(manifest, mesh)'''),nbformat.v4.new_code_cell('''display(Image(filename=str(room_upgrade.OUT / 'detector_comparison_contact.png')))
print('Matched frames and thresholds; detections are qualitative evidence, not mAP.')'''),nbformat.v4.new_code_cell('''display(Image(filename=str(room_upgrade.OUT / 'room_floorplan.png')))
display(Image(filename=str(room_upgrade.OUT / 'room_topview_rgb.png')))'''),nbformat.v4.new_code_cell('''display(Image(filename=str(room_upgrade.OUT / 'room_3d_cutaway.png')))
display(HTML('<a href="outputs/room_refined/room_3d_interactive.html" target="_blank">Open standalone interactive 3D: cutaway / full room / objects</a>'))''')]
n.metadata['kernelspec']={'display_name':'Python (local mapping)','language':'python','name':'python3'}
nbformat.write(n,root/'notebookgyu-cpu.room26.ipynb')
from nbclient import NotebookClient
client=NotebookClient(n,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(root)}})
# Explicit interpreter avoids the unrelated system Python on PATH.
from jupyter_client import KernelManager
km=KernelManager(kernel_name='python3')
km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
client.km=km
print('Starting pinned kernel',km.kernel_spec.argv,flush=True)
client.execute()
nbformat.write(n,root/'notebookgyu-cpu.room26.executed.ipynb')
print('Executed',len([c for c in n.cells if c.cell_type=='code']),'code cells')
