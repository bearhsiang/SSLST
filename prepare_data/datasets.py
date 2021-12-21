from pathlib import Path
import importlib

datasets = {}

search_root = Path(__file__).parent

for file in search_root.glob("*.py"):
    name = file.stem

    try:
        module_name = f'{name}'
        module = importlib.import_module(module_name, package=__package__)
        # if hasattr(module, 'Dataset'):
        #     datasets[module.Dataset.get_name()] = module.Dataset
        # TODO: print some error messages here
        try:
            datasets[module.Dataset.get_name()] = module.Dataset
        except:
            pass

    except ModuleNotFoundError as e:
        full_package = f"{__package__}{module_name}"
        print(f'[{__name__}] Warning: can not import {full_package}: {str(e)}. Pass.')
        continue