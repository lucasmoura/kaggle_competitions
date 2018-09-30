from setuptools import setup, find_packages

setup(name='kaggleflow',
      version='0.1',
      description='Kaggle helper for competitions',
      classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3'
      ],
      url='https://github.com/lucasmoura/kaggle_competitions/',
      author='Lucas Moura',
      author_email='lucas.moura128@gmail.com',
      license='GPL',
      packages=find_packages(),
      install_requires=[
        'numpy',
        'pandas'
      ],
      entry_points={
        'console_scripts': ['kaggleflow=kaggleflow.cli.runner:runner']
      },
      test_suite='tests',
      zip_safe=False)
