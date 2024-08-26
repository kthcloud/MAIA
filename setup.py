import os

import setuptools
from setuptools import setup

import versioneer


def resolve_requirements(file):
    requirements = []
    with open(file) as f:
        req = f.read().splitlines()
        for r in req:
            if r.startswith("-r"):
                requirements += resolve_requirements(os.path.join(os.path.dirname(file), r.split(" ")[1]))
            else:
                requirements.append(r)
    return requirements

def read_file(file):
    with open(file) as f:
        content = f.read()
    return content


setup(
    version=versioneer.get_version(),
    packages=setuptools.find_packages(),
    cmdclass=versioneer.get_cmdclass(),
    zip_safe=False,
    data_files=[('', ["requirements.txt"]), ],
    package_data={
        "": ["configs/*.json"],
    },
    entry_points={
        "console_scripts": [
            "MAIAKubeGate_deploy_helm_chart = MAIAKubeGate_scripts.MAIAKubeGate_deploy_helm_chart:main",
            "MAIAKubeGate_deploy_MAIA_namespace = MAIAKubeGate_scripts.MAIAKubeGate_deploy_MAIA_namespace:main",
            "MAIAKubeGate_create_JupyterHub_config = MAIAKubeGate_scripts.MAIAKubeGate_create_JupyterHub_config:main",
            "MAIAKubeGate_create_MAIA_Addons_config =  MAIAKubeGate_scripts.MAIAKubeGate_create_MAIA_Addons_config:main",
        ],
    },
    keywords=["helm", "kubernetes", "maia", "resource deployment"],

)
