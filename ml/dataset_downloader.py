"""Descarga y organiza múltiples datasets públicos de enfermedades de plantas."""Descarga y organiza múltiples datasets públicos de enfermedades de plantas.



Datasets objetivo (se requiere aceptación de licencias manual en algunos casos):Datasets objetivo (se requiere aceptación de licencias manual en algunos casos):

- PlantVillage (https://www.kaggle.com/datasets/emmarex/plantdisease) (requiere credenciales Kaggle)- PlantVillage (https://www.kaggle.com/datasets/emmarex/plantdisease) (requiere credenciales Kaggle)

- PlantDoc (https://github.com/pratikkayal/PlantDoc-Dataset)- PlantDoc (https://github.com/pratikkayal/PlantDoc-Dataset)

- Cassava Leaf Disease (Kaggle)- Cassava Leaf Disease (Kaggle)

- AI Challenger (subset plantas si disponible)- AI Challenger (subset plantas si disponible)



Este script provee funciones stub y guía; algunas descargas requieren API de Kaggle configurada (~/.kaggle/kaggle.json).Este script provee funciones stub y guía; algunas descargas requieren API de Kaggle configurada (~/.kaggle/kaggle.json).

""""""

from pathlib import Pathfrom pathlib import Path

import subprocessimport subprocess

import shutilimport shutil

import tarfileimport tarfile

import zipfileimport zipfile

import jsonimport json

import osimport os

import sysimport sys

import timeimport time

from typing import Listfrom typing import List



DATASETS = {DATASETS = {

    'plantvillage': {    'plantvillage': {

        'kaggle': 'emmarex/plantdisease',        'kaggle': 'emmarex/plantdisease',

        'mapping_rule': 'folder_names',        'mapping_rule': 'folder_names',

        'desc': 'PlantVillage plant disease classification'        'desc': 'PlantVillage plant disease classification'

    },    },

    'cassava': {    'cassava': {

        'kaggle': 'akash2907/cassava-disease-dataset',        'kaggle': 'akash2907/cassava-disease-dataset',

        'mapping_rule': 'csv_labels',        'mapping_rule': 'csv_labels',

        'desc': 'Cassava Leaf Disease'        'desc': 'Cassava Leaf Disease'

    },    },

    'plantdoc': {    'plantdoc': {

        'git': 'https://github.com/pratikkayal/PlantDoc-Dataset',        'git': 'https://github.com/pratikkayal/PlantDoc-Dataset',

        'mapping_rule': 'folder_names',        'mapping_rule': 'folder_names',

        'desc': 'PlantDoc field images'        'desc': 'PlantDoc field images'

    }    }

}}



TARGET_SPECIES_MAP = {TARGET_SPECIES_MAP = {

    # Ejemplo de normalización: ('tomato', 'Tomato___Bacterial_spot') -> species='tomato', disease='bacterial_spot'    # Ejemplo de normalización: ('tomato', 'Tomato___Bacterial_spot') -> species='tomato', disease='bacterial_spot'

}}



def ensure_dir(p: Path):

    p.mkdir(parents=True, exist_ok=True)def ensure_dir(p: Path):

    p.mkdir(parents=True, exist_ok=True)

def _have_cli(cmd: str) -> bool:

    return shutil.which(cmd) is not None

def _have_cli(cmd: str) -> bool:

def _extract_all_in_dir(dest: Path):    return shutil.which(cmd) is not None

    changed = True

    while changed:

        changed = Falsedef _extract_all_in_dir(dest: Path):

        for z in list(dest.rglob('*.zip')):    changed = True

            print(f"  > Extrayendo {z.relative_to(dest)}")    # Repetir mientras sigan apareciendo archivos comprimidos anidados

            try:    while changed:

                with zipfile.ZipFile(z, 'r') as f:        changed = False

                    f.extractall(z.parent)        for z in list(dest.rglob('*.zip')):

                z.unlink()            print(f"  > Extrayendo {z.relative_to(dest)}")

                changed = True            try:

            except Exception as e:                with zipfile.ZipFile(z, 'r') as f:

                print(f"  ! Fallo extrayendo {z}: {e}")                    f.extractall(z.parent)

        for t in list(dest.rglob('*.tar')) + list(dest.rglob('*.tar.gz')) + list(dest.rglob('*.tgz')):                z.unlink()

            print(f"  > Extrayendo {t.relative_to(dest)}")                changed = True

            try:            except Exception as e:

                with tarfile.open(t, 'r:*') as f:                print(f"  ! Fallo extrayendo {z}: {e}")

                    f.extractall(t.parent)        for t in list(dest.rglob('*.tar')) + list(dest.rglob('*.tar.gz')) + list(dest.rglob('*.tgz')):

                t.unlink()            print(f"  > Extrayendo {t.relative_to(dest)}")

                changed = True            try:

            except Exception as e:                with tarfile.open(t, 'r:*') as f:

                print(f"  ! Fallo extrayendo {t}: {e}")                    f.extractall(t.parent)

                t.unlink()

def download_kaggle_dataset(ref: str, dest: Path):                changed = True

    dest.mkdir(parents=True, exist_ok=True)            except Exception as e:

    print(f"Descargando {ref} ...")                print(f"  ! Fallo extrayendo {t}: {e}")

    if _have_cli('kaggle'):

        try:

            subprocess.run(["kaggle", "datasets", "download", "-d", ref, "-p", str(dest)], check=True)def download_kaggle_dataset(ref: str, dest: Path):

        except subprocess.CalledProcessError as e:    dest.mkdir(parents=True, exist_ok=True)

            raise RuntimeError(f"Fallo CLI kaggle: {e}")    print(f"Descargando {ref} ...")

    else:    if _have_cli('kaggle'):

        try:        try:

            from kaggle.api.kaggle_api_extended import KaggleApi            subprocess.run(["kaggle", "datasets", "download", "-d", ref, "-p", str(dest)], check=True)

        except ImportError:        except subprocess.CalledProcessError as e:

            raise RuntimeError("No se encontró comando 'kaggle' ni librería Python. Instala con: pip install kaggle")            raise RuntimeError(f"Fallo CLI kaggle: {e}")

        api = KaggleApi()    else:

        try:        # Fallback API Python

            api.authenticate()        try:

        except Exception as e:            from kaggle.api.kaggle_api_extended import KaggleApi

            raise RuntimeError(f"Autenticación Kaggle falló: {e}. Asegúrate de tener kaggle.json en %USERPROFILE%/.kaggle")        except ImportError:

        api.dataset_download_files(ref, path=str(dest), quiet=False, unzip=False)            raise RuntimeError("No se encontró comando 'kaggle' ni librería Python. Instala con: pip install kaggle")

    _extract_all_in_dir(dest)        api = KaggleApi()

        try:

def clone_git_dataset(url: str, dest: Path):            api.authenticate()

    if dest.exists() and any(dest.iterdir()):        except Exception as e:

        print(f"Destino {dest} no vacío, omitiendo clone.")            raise RuntimeError(f"Autenticación Kaggle falló: {e}. Asegúrate de tener kaggle.json en %USERPROFILE%/.kaggle")

        return        ref_user, ref_ds = ref.split('/')

    if not _have_cli('git'):        api.dataset_download_files(ref, path=str(dest), quiet=False, unzip=False)

        raise RuntimeError("'git' no está disponible en PATH, clona manualmente: git clone {url} {dest}")    _extract_all_in_dir(dest)

    print(f"Clonando {url} ...")

    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)

def clone_git_dataset(url: str, dest: Path):

def list_available():    if dest.exists() and any(dest.iterdir()):

    print("Datasets disponibles:")        print(f"Destino {dest} no vacío, omitiendo clone.")

    for k, meta in DATASETS.items():        return

        src = meta.get('kaggle') or meta.get('git')    if not _have_cli('git'):

        print(f" - {k:10s} : {meta['desc']}  -> {src}")        raise RuntimeError("'git' no está disponible en PATH, clona manualmente: git clone {url} {dest}")

    print(f"Clonando {url} ...")

def validate_kaggle_config():    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)

    home = Path.home()

    cfg = home / '.kaggle' / 'kaggle.json'

    if not cfg.exists():def list_available():

        print("[AVISO] No se encontró kaggle.json en:", cfg)    print("Datasets disponibles:")

        print("Crea el archivo con tu API Token (Cuenta Kaggle -> Create New API Token).")    for k, meta in DATASETS.items():

        return False        src = meta.get('kaggle') or meta.get('git')

    return True        print(f" - {k:10s} : {meta['desc']}  -> {src}")



def main():

    import argparsedef validate_kaggle_config():

    parser = argparse.ArgumentParser(description='Descarga y prepara datasets de enfermedades de plantas')    # En Windows: %USERPROFILE%/.kaggle/kaggle.json

    parser.add_argument('--out', type=str, default='combined_data', help='Directorio raíz de salida')    home = Path.home()

    parser.add_argument('--datasets', type=str, nargs='*', default=['plantvillage'], help='Lista de datasets a descargar')    cfg = home / '.kaggle' / 'kaggle.json'

    parser.add_argument('--list', action='store_true', help='Listar datasets disponibles y salir')    if not cfg.exists():

    parser.add_argument('--skip-kaggle-check', action='store_true')        print("[AVISO] No se encontró kaggle.json en:", cfg)

    parser.add_argument('--fail-fast', action='store_true', help='Detener en primer error')        print("Crea el archivo con tu API Token (Cuenta Kaggle -> Create New API Token).")

    args = parser.parse_args()        return False

    return True

    if args.list:

        list_available()

        returndef main():

    import argparse

    out = Path(args.out); ensure_dir(out)    parser = argparse.ArgumentParser(description='Descarga y prepara datasets de enfermedades de plantas')

    if not args.skip_kaggle_check:    parser.add_argument('--out', type=str, default='combined_data', help='Directorio raíz de salida')

        validate_kaggle_config()    parser.add_argument('--datasets', type=str, nargs='*', default=['plantvillage'], help='Lista de datasets a descargar')

    parser.add_argument('--list', action='store_true', help='Listar datasets disponibles y salir')

    errors = []    parser.add_argument('--skip-kaggle-check', action='store_true')

    for ds in args.datasets:    parser.add_argument('--fail-fast', action='store_true', help='Detener en primer error')

        if ds not in DATASETS:    args = parser.parse_args()

            print(f"[WARN] Dataset desconocido: {ds}")

            continue    if args.list:

        meta = DATASETS[ds]        list_available()

        print(f"==> Procesando {ds} - {meta['desc']}")        return

        start = time.time()

        dest = out / f"{ds}_raw"    out = Path(args.out); ensure_dir(out)

        try:    if not args.skip_kaggle_check:

            if 'kaggle' in meta:        validate_kaggle_config()

                download_kaggle_dataset(meta['kaggle'], dest)

            elif 'git' in meta:    errors = []

                clone_git_dataset(meta['git'], dest)    for ds in args.datasets:

            else:        if ds not in DATASETS:

                print(f"[WARN] Sin método de descarga para {ds}")            print(f"[WARN] Dataset desconocido: {ds}")

                continue            continue

            print(f"    ✓ Completado {ds} en {time.time()-start:.1f}s")        meta = DATASETS[ds]

        except Exception as e:        print(f"==> Procesando {ds} - {meta['desc']}")

            msg = f"    ✗ Error descargando {ds}: {e}"        start = time.time()

            print(msg)        dest = out / f"{ds}_raw"

            errors.append(msg)        try:

            if args.fail_fast:            if 'kaggle' in meta:

                break                download_kaggle_dataset(meta['kaggle'], dest)

            elif 'git' in meta:

    if errors:                clone_git_dataset(meta['git'], dest)

        print('\nResumen de errores:')            else:

        for e in errors:                print(f"[WARN] Sin método de descarga para {ds}")

            print(' -', e)                continue

    else:            print(f"    ✓ Completado {ds} en {time.time()-start:.1f}s")

        print('\nTodos los datasets solicitados descargados correctamente.')        except Exception as e:

            msg = f"    ✗ Error descargando {ds}: {e}"

    print('\nSiguiente paso: normalizar clases -> python ml/class_normalizer.py --src combined_data --dst ml/data/clean')            print(msg)

            errors.append(msg)

if __name__ == '__main__':            if args.fail_fast:

    main()                break


    if errors:
        print('\nResumen de errores:')
        for e in errors:
            print(' -', e)
    else:
        print('\nTodos los datasets solicitados descargados correctamente.')

    print('\nSiguiente paso: normalizar clases -> python ml/class_normalizer.py --src combined_data --dst ml/data/clean')

if __name__ == '__main__':
    main()
