"""Refuse unreviewed changes to original model/data/metric source files."""
import hashlib
import json
from pathlib import Path


def verify_original(project_root):
    project_root = Path(project_root)
    manifest = json.loads(Path(__file__).with_name('source_manifest.json').read_text(encoding='utf-8'))
    changed = []
    for relative, expected in manifest.items():
        path = project_root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            changed.append(relative)
    if changed:
        raise RuntimeError('Original source differs from reviewed 17474 version: ' + ', '.join(changed))
    print('ORIGINAL_SOURCE_CHECK_OK', len(manifest))
