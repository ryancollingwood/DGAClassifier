import os.path
from os import getcwd
import PyInstaller.__main__


def package_model(model_file_path: str, package_name: str):
    working_dir = getcwd()
    print(f"Current working directory: {working_dir}")

    sklearn_path = '.venv/Lib/site-packages/sklearn/.libs/'

    abs_vcomp_path = os.path.join(working_dir, sklearn_path, 'vcomp140.dll').replace("\\", "/")
    abs_model_path = os.path.join(working_dir, model_file_path).replace("\\", "/")
    abs_ico_path = os.path.join(working_dir, 'resources', 'icon.ico').replace("\\", "/")
    #entry_script = os.path.join(working_dir, 'dga_classify.py').replace("\\", "/")
    entry_script = os.path.join("dga_classify.py")

    print("abs_vcomp_path", abs_vcomp_path)
    print("abs_model_path", abs_model_path)
    print("abs_ico_path", abs_ico_path)

    hidden_imports = [
        'pandas', ' sklearn', ' sklearn.utils.sparsetools._graph_validation',
        'sklearn.utils.sparsetools._graph_tools', 'sklearn.utils.lgamma',
        'sklearn.neighbors._typedefs', 'sklearn.utils._cython_blas',
        'sklearn.neighbors._quad_tree', 'sklearn.tree._utils',
        'sklearn.neighbors._typedefs',
    ]

    hidden_import_cmd = " ".join([
        f"--hidden-import {x}" for x in hidden_imports
    ])


    PyInstaller.__main__.run([
        f'{entry_script}',
        '--name=%s' % package_name,
        '--onefile',
        '--console',
        f'--add-binary={abs_vcomp_path};.',
        f'--add-data={abs_model_path};models',
        #f'{hidden_import_cmd}',
        f'--icon={abs_ico_path}',
    ])


if __name__ == "__main__":
    package_model("integrationtests/models/trained.model", "dummy")

