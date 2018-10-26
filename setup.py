from distutils.core import setup

setup(name='pke',
      version='1.72',
      description='Python Keyphrase Extraction module',
      author='Florian Boudin',
      author_email='florian.boudin@univ-nantes.fr',
      license='gnu',
      packages=['pke', 'pke.unsupervised', 'pke.supervised',
                'pke.supervised.feature_based', 'pke.unsupervised.graph_based',
                'pke.unsupervised.statistical'],
      url="https://github.com/boudinfl/pke",
      install_requires=[
          'nltk',
          'networkx',
          'numpy',
          'scipy',
          'sklearn',
          'unidecode',
          'future'
      ],
      package_data={'pke': ['models/*.pickle', 'models/*.gz']}
      )
