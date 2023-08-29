from setuptools import setup, find_packages

setup(
    name='latent',
    version='0.1.0',
    packages=find_packages(),
    author='EleutherAI',
    author_email='contact@eleuther.ai',
    description='Latent video diffusion training and generation.',
    url='https://github.com/EleutherAI/latentvideo',
    license='MIT',
    install_requires=[
        'numpy',
        'cv2',
        'equinox',
        'jax',
        # Other dependencies
    ],
    entry_points={
        'console_scripts': [
            'lvd = scripts.main:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
    ],
)
