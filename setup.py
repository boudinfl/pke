from distutils.core import setup

setup(name='pke',
      version='1.0',
      description='Python Keyphrase Extraction module',
      author='Florian Boudin',
      author_email='florian.boudin@univ-nantes.fr',
      packages=['pke'],
      url="https://github.com/boudinfl/pke",
      install_requires=[
      	'corenlp_parser',
      	'nltk',
      	'networkx',
      	'numpy',
      	'scipy'
      ],
      dependency_links=[
        'https://github.com/boudinfl/corenlp_parser/archive/master.zip'
      ]
     )

