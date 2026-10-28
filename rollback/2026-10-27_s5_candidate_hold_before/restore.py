from pathlib import Path
import json, shutil, hashlib
from datetime import datetime
base = Path(__file__).resolve().parent
root = base.parent.parent
manifest = json.loads((base / 'manifest.json').read_text())
for name in manifest['restore_files']:
    assert hashlib.sha256((base / name).read_bytes()).hexdigest() == manifest['before_sha256'][name], name
saved = base / ('preserved_before_restore_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
for name in manifest['restore_files'] + manifest['new_files']:
    current = root / name
    if current.exists():
        (saved / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(current, saved / name)
for name in manifest['restore_files']:
    shutil.copy2(base / name, root / name)
for name in manifest['new_files']:
    (root / name).unlink(missing_ok=True)
print('Restored. Subsequent changes preserved at:', saved)
