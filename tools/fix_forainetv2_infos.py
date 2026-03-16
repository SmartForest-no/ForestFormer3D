#!/usr/bin/env python3
"""
Fix ForAINetV2 info pkl files: convert list -> dict structure expected by dataset.

Usage:
  - Non-root: creates `*.fixed.pkl` alongside originals and prints commands to replace files.
  - Root: creates backups `*.bak` and overwrites originals in-place.
"""
import argparse
import os
import pickle
import shutil
import sys

DEFAULT_ROOT = os.path.join('data', 'ForAINetV2')

def process_file(p):
    with open(p,'rb') as f:
        data = pickle.load(f)
    if isinstance(data, (list,tuple)):
        new = {'metainfo': {}, 'data_list': list(data)}
        return new
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root',
        default=DEFAULT_ROOT,
        help='Path to ForAINetV2 dataset root (default: data/ForAINetV2).'
    )
    args = parser.parse_args()

    root = args.root
    if not os.path.isdir(root):
        print('Root folder not found:', root)
        sys.exit(1)
    files = [os.path.join(root, f) for f in os.listdir(root)
             if f.startswith('forainetv2_oneformer3d_infos') and f.endswith('.pkl')]
    if not files:
        print('No info pkl files found in', ROOT)
        return
    uid = os.geteuid()
    can_overwrite = (uid == 0)
    replacements = []
    for p in files:
        try:
            with open(p,'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            print('Failed to read', p, e)
            continue
        if isinstance(data, (list,tuple)):
            new = {'metainfo': {}, 'data_list': list(data)}
            if can_overwrite:
                bak = p + '.bak'
                shutil.copy(p, bak)
                with open(p,'wb') as f:
                    pickle.dump(new, f)
                print('Overwrote', p, 'with backup', bak)
            else:
                outp = p + '.fixed.pkl'
                with open(outp,'wb') as f:
                    pickle.dump(new, f)
                print('Wrote fixed file:', outp)
                replacements.append((outp, p))
        else:
            print('No change needed for', p)

    if replacements:
        print('\nRun the following commands as root (or move files manually) to apply fixes:')
        for outp, orig in replacements:
            print('  mv', outp, orig)
        print('\nOr run: sudo python3 tools/fix_forainetv2_infos.py')

if __name__ == '__main__':
    main()
