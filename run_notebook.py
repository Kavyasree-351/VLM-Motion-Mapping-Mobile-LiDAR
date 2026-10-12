import os,time,json
from pathlib import Path
import nbformat
from nbclient import NotebookClient
root=Path(__file__).resolve().parent
os.environ['PYTHONPATH']=str(root/'.packages')
os.environ['YOLO_CONFIG_DIR']=str(root/'.yolo')
os.environ['MPLCONFIGDIR']=str(root/'.matplotlib')
import imageio_ffmpeg
os.environ['IMAGEIO_FFMPEG_EXE']=imageio_ffmpeg.get_ffmpeg_exe()
n=nbformat.read(root/'notebookgyu-cpu.local.ipynb',as_version=4)
n.cells.insert(0,nbformat.v4.new_code_cell('import matplotlib as mpl\nmpl.rcParams["animation.ffmpeg_path"] = '+repr(imageio_ffmpeg.get_ffmpeg_exe())))
def save(): nbformat.write(n,root/'notebookgyu-cpu.executed.ipynb')
def start(cell,cell_index,**kw):
 print(f'START cell {cell_index-1}',flush=True)
def done(cell,cell_index,**kw):
 errors=[o for o in cell.get('outputs',[]) if o.output_type=='error']
 print(f'END cell {cell_index-1} '+ ('ERROR '+errors[0]['ename']+': '+errors[0]['evalue'] if errors else 'OK'),flush=True)
 save()
client=NotebookClient(n,timeout=None,kernel_name='python3',allow_errors=True,resources={'metadata':{'path':str(root)}},on_cell_start=start,on_cell_executed=done)
try: client.execute()
finally: save()
errors=[{'cell':i-1,'name':o.ename,'message':o.evalue} for i,c in enumerate(n.cells) for o in c.get('outputs',[]) if o.output_type=='error']
(root/'execution_errors.json').write_text(json.dumps(errors,indent=2),encoding='utf-8')
print('COMPLETE; errors:',len(errors),flush=True)
