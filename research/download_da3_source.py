from pathlib import Path
import urllib.request, zipfile, io
root=Path('D:/VLM-maps/.models/da3-source');root.mkdir(parents=True,exist_ok=True)
blob=urllib.request.urlopen('https://codeload.github.com/ByteDance-Seed/Depth-Anything-3/zip/refs/heads/main',timeout=90).read()
z=zipfile.ZipFile(io.BytesIO(blob))
for item in z.infolist():
    target=(root/item.filename).resolve()
    if not target.is_relative_to(root.resolve()): raise ValueError('Archive path outside source directory')
z.extractall(root)
print('Official source downloaded',len(blob))
