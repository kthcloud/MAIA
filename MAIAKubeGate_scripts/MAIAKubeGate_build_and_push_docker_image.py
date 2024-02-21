#!/usr/bin/env python

import datetime
import json
import os
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

import yaml

import MAIAKubeGate
from MAIAKubeGate.kaniko import build_and_upload_image_on_docker_registry

version = MAIAKubeGate.__version__

TIMESTAMP = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())

DESC = dedent(
    """
    Script to Build and Deploy a Docker Image on a Private Docker Registry.
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename}  --config-file
    """.format(  # noqa: E501
        filename=Path(__file__).stem
    )
)


def get_arg_parser():
    pars = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)

    pars.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="JSON configuration file used to generate the custom values used in the maiakubegate-kaniko and hive-deploy Helm charts.",
    )

    pars.add_argument('-v', '--version', action='version', version='%(prog)s ' + version)

    return pars


def main():
    parser = get_arg_parser()

    arguments = vars(parser.parse_args())

    config_file = arguments["config_file"]

    with open(config_file) as json_file:
        config_dict = json.load(json_file)

    kubeconfig = yaml.safe_load(Path(os.environ["KUBECONFIG"]).read_text())

    config_dict["docker_image"] = build_and_upload_image_on_docker_registry(config_dict, kubeconfig,
                                                                            "registry.maia.cloud.cbh.kth.se",
                                                                            interactive=True)

    print("Docker Image successfully pushed at : {}".format(config_dict["docker_image"]))


if __name__ == "__main__":
    main()
