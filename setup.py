from distutils.core import setup

setup(
    name='transformers_ptuning',
    version='1.0',
    description='P-tuning wrapper for huggingface transformers',
    author='Wanicca',
    author_email='wanicca@gmail.com',
    url='https://github.com/wanicca/transformers_ptuning/',
    packages=['transformers_ptuning'],
    requires=['transformers>=4.0.0']
)