#!/usr/bin/env python3
"""
Script to generate noise-injected configuration files.
Reads all YAML files from configs/full_scale/ and creates three variants
(low, mid, high noise) for each in configs/config_noise/.
"""

import yaml
from pathlib import Path

# Define noise levels based on the approved plan
NOISE_LEVELS = {
    'low': {
        'depolarizing_p1': 0.0005,
        'depolarizing_p2': 0.005,
    },
    'mid': {
        'depolarizing_p1': 0.001,
        'depolarizing_p2': 0.01,
    },
    'high': {
        'depolarizing_p1': 0.005,
        'depolarizing_p2': 0.05,
    }
}

def main():
    source_dir = Path('configs/full_scale')
    target_dir = Path('configs/config_noise')

    # Ensure target directory exists
    target_dir.mkdir(parents=True, exist_ok=True)

    # Get all YAML files from source directory
    source_files = sorted(source_dir.glob('*.yaml'))

    print(f"Found {len(source_files)} source configuration files.")

    total_generated = 0

    for source_file in source_files:
        # Read the source YAML
        with open(source_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Extract base name (remove _full.yaml suffix if present)
        base_name = source_file.stem
        if base_name.endswith('_full'):
            base_name = base_name[:-5]  # Remove '_full' suffix

        # Generate three noise variants
        for noise_level, noise_params in NOISE_LEVELS.items():
            # Create a copy of the config
            noisy_config = config.copy()

            # Update experiment name
            original_exp_name = noisy_config.get('experiment_name', base_name)
            noisy_config['experiment_name'] = f"{base_name}_noise_{noise_level}"

            # Ensure environment section exists
            if 'environment' not in noisy_config:
                noisy_config['environment'] = {}

            # Enable noise and set parameters
            noisy_config['environment']['add_noise'] = True
            noisy_config['environment']['noise'] = noise_params

            # Generate output filename
            output_filename = f"{base_name}_noise_{noise_level}.yaml"
            output_path = target_dir / output_filename

            # Write the new YAML file
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(noisy_config, f, default_flow_style=False, sort_keys=False)

            print(f"  ✓ Created: {output_filename}")
            total_generated += 1

    print(f"\n✅ Successfully generated {total_generated} noise configuration files in {target_dir}/")

    # Verify the count
    if total_generated == 18:
        print("✅ Verification passed: Exactly 18 files generated as expected.")
    else:
        print(f"⚠️  Warning: Expected 18 files, but generated {total_generated}.")

if __name__ == '__main__':
    main()
