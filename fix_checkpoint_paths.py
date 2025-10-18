#!/usr/bin/env python3
"""
fix_checkpoint_paths.py

- Iterates through all YAML files in `configs/config_noise/`.
- Sets a unique `checkpoint_dir` and `checkpoint_prefix` for each file
  to prevent experiment results from overwriting each other.
- The new path is based on the experiment name, e.g.,
  `checkpoints/<dataset>_<encoding>/noise_<level>`
"""
import yaml
from pathlib import Path
import re

def main():
    config_dir = Path('configs/config_noise')
    if not config_dir.is_dir():
        print(f"❌ Error: Directory not found: {config_dir}")
        return

    yaml_files = sorted(config_dir.glob('*.yaml'))

    print(f"Found {len(yaml_files)} config files to process in {config_dir}/\n")

    total_modified = 0
    for file_path in yaml_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # --- Get key components from filename ---
            # Filename example: mnist_amplitude_noise_low.yaml
            match = re.match(
                r'(mnist|fashion)_([a-zA-Z]+)_noise_(low|mid|high)',
                file_path.stem
            )

            if not match:
                print(f"⚠️  Skipping non-standard file: {file_path.name}")
                continue

            dataset, encoding, noise_level = match.groups()
            prefix_dataset = 'fashion' if dataset == 'fashion' else 'mnist'
            prefix_encoding = 'amp' if encoding == 'amplitude' else encoding

            # --- Define new unique paths ---
            base_dir = f"{dataset}_{encoding}"
            new_dir = f'checkpoints/{base_dir}/noise_{noise_level}'
            new_prefix = f'{prefix_dataset}_{prefix_encoding}_noise_{noise_level}'

            # --- Check if modification is needed ---
            current_dir = config.get('training', {}).get('checkpoint_dir', '')

            if current_dir == new_dir:
                print(f"- Skipping '{file_path.name}', already correct.")
                continue

            # --- Apply modifications ---
            if 'training' not in config:
                config['training'] = {}

            config['training']['checkpoint_dir'] = new_dir
            config['training']['checkpoint_prefix'] = new_prefix

            # --- Write back to the file ---
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            print(f"✅ Updated '{file_path.name}':")
            print(f"   - dir: {new_dir}")
            print(f"   - prefix: {new_prefix}")
            total_modified += 1

        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")

    print(f"\n🎉 Successfully updated {total_modified} configuration files.")

if __name__ == '__main__':
    main()
