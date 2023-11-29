from setuptools import setup

with open("src/turnkeyml/version.py", encoding="utf-8") as fp:
    version = fp.read().split('"')[1]

setup(
    name="turnkeyml",
    version=version,
    description="TurnkeyML Tools and Models",
    author="Jeremy Fowers, Daniel Holanda, Ramakrishnan Sivakumar, Victoria Godsoe",
    author_email="jeremy.fowers@amd.com, daniel.holandanoronha@amd.com, krishna.sivakumar@amd.com, victoria.godsoe@amd.com",
    package_dir={"": "src", "turnkeyml_models": "models"},
    packages=[
        "turnkeyml",
        "turnkeyml.analyze",
        "turnkeyml.build",
        "turnkeyml.run",
        "turnkeyml.run.onnxrt",
        "turnkeyml.run.tensorrt",
        "turnkeyml.run.torchrt",
        "turnkeyml.cli",
        "turnkeyml.common",
        "turnkeyml_models",
        "turnkeyml_models.diffusers",
        "turnkeyml_models.graph_convolutions",
        "turnkeyml_models.llm",
        "turnkeyml_models.llm_layer",
        "turnkeyml_models.popular_on_huggingface",
        "turnkeyml_models.selftest",
        "turnkeyml_models.timm",
        "turnkeyml_models.torch_hub",
        "turnkeyml_models.torchvision",
        "turnkeyml_models.transformers",
    ],
    install_requires=[
        "invoke>=2.0.0",
        "onnx>=1.11.0",
        "onnxmltools==1.10.0",
        "hummingbird-ml==0.4.4",
        "scikit-learn==1.1.1",
        "xgboost==1.6.1",
        "lightgbm==3.3.5",
        "onnxruntime >=1.10.1, <1.16.0",
        "paramiko==2.11.0",
        "torch>=1.12.1",
        "protobuf>=3.17.3,<3.21",
        "pyyaml>=5.4",
        "typeguard>=2.3.13",
        "packaging>=20.9",
        "pandas>=1.5.3",
        "fasteners",
        "GitPython>=3.1.40",
    ],
    extras_require={
        "tensorflow": [
            "tensorflow-cpu==2.8.1",
            "tf2onnx>=1.12.0",
        ],
    },
    classifiers=[],
    entry_points={
        "console_scripts": [
            "turnkey=turnkeyml:turnkeycli",
        ]
    },
    python_requires=">=3.8, <3.11",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    package_data={
        "turnkeyml.api": ["Dockerfile"],
        "turnkeyml_models": ["requirements.txt", "readme.md"],
    },
)
