from distutils.core import setup

setup(name='pke',
      version='1.04',
      description='Python Keyphrase Extraction module',
      author='Florian Boudin',
      author_email='florian.boudin@univ-nantes.fr',
      license='gnu',
      packages=['pke'],
      url="https://github.com/boudinfl/pke",
      install_requires=[
      	'nltk',
      	'networkx',
      	'numpy',
      	'scipy',
            'sklearn'
      ],
      package_data={'pke': ['models/*.pickle']}
     )

