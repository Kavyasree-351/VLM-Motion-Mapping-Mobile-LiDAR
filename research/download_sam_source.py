from pathlib import Path
import urllib.request,zipfile,io
root=Path('D:/VLM-maps/.models/sam3-source');root.mkdir(parents=True,exist_ok=True)
blob=urllib.request.urlopen('https://codeload.github.com/facebookresearch/sam3/zip/refs/heads/main',timeout=90).read();z=zipfile.ZipFile(io.BytesIO(blob))
for item in z.infolist():
 if not (root/item.filename).resolve().is_relative_to(root.resolve()):raise ValueError('Invalid archive path')
z.extractall(root);print('Official SAM source downloaded:',len(blob))
