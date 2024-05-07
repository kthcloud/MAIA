import setuptools
from setuptools import setup

import versioneer


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
        "": ["configs/*.yml", "configs/*.json"],
    },
    entry_points={
        "console_scripts": [
            "MAIAKubeGate_deploy_helm_chart = MAIAKubeGate_scripts.MAIAKubeGate_deploy_helm_chart:main",
            "MAIAKubeGate_build_and_push_docker_image = MAIAKubeGate_scripts.MAIAKubeGate_build_and_push_docker_image:main"
        ],
    },
    keywords=["helm", "kubernetes", "maia", "resource deployment"],

)
