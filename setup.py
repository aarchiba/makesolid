from setuptools import setup, find_packages

def readme():
    return open('README.md').read()

setup(
    name="makesolid",
    version="0.1",
    description="Tools for designing 3D objects from Python",
    long_description=readme(),
    url="https://github.com/aarchiba/makesolid",
    author="Anne Archibald",
    author_email="peridot.faceted@gmail.com",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'solidpython',
        ],
    include_package_data=True,
    zip_safe=False,
)

