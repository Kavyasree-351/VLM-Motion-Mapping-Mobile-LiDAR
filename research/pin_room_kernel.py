from pathlib import Path
p=Path('research/build_room_notebook.py');s=p.read_text();s=s.replace("client.execute(kernel_cmd=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'])", "from jupyter_client import KernelManager\nkm=KernelManager(kernel_name='python3')\nkm.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']\nclient.km=km\nprint('Starting pinned kernel',km.kernel_spec.argv,flush=True)\nclient.execute()")
p.write_text(s)
