from attack import main

import glob
import os
import sys

assert len(sys.argv) == 3, "argument not found. run like: python attack_loop.py model_folder/ True"
model_root_folder = sys.argv[1]
model_is_retrofitted = sys.argv[2]
if model_is_retrofitted == 'True':
    model_is_retrofitted = True
elif model_is_retrofitted == 'False':
    model_is_retrofitted = False

print('getting models from path:', model_root_folder)
models = glob.glob(os.path.join(model_root_folder, 'save_epoch_*.pth'))
print(f'got {len(models)} models:', models)
for model_path in models:
    print('attacking model:', model_path)
    main(model_path, model_is_retrofitted)
    print()
    print()