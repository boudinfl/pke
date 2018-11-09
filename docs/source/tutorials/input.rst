Input formats
=============

pke currently supports the following input formats:

1. *raw text*: text pre-processing (i.e. tokenization, sentence splitting and
   POS-tagging) is carried out using `spacy <https://spacy.io/>`_.

   Example of content from a raw text file:

   .. code::

      Efficient discovery of grid services is essential for the success of
      grid computing. [...]

   To read a document in raw text format:

   .. code::

      extractor = pke.unsupervised.TopicRank()
      extractor.load_document(input='/path/to/input.txt', language='en')

2. *input text*: same as raw text, text pre-processing is carried out using
   spacy.

   To read an input text:

   .. code::

     extractor = pke.unsupervised.TopicRank()
     text = u'Efficient discovery of grid services is essential for the [...]'
     extractor.load_document(input=text, language='en')


3. *Stanford XML CoreNLP*: output file produced using the annotators `tokenize`,
   `ssplit`, `pos` and `lemma`. Document logical structure information can by
   specified by incorporating attributes into the sentence elements of the
   CoreNLP XML format.

   Example of CoreNLP XML:

   .. code::

     <?xml version="1.0" encoding="UTF-8"?>
     <root>
       <document>
         <sentences>
           <sentence id="1" section="abstract" type="bodyText" confidence="0.925">
            <tokens>
               <token id="1">
                 <word>Efficient</word>
                 <lemma>efficient</lemma>
                 <CharacterOffsetBegin>362</CharacterOffsetBegin>
                 <CharacterOffsetEnd>371</CharacterOffsetEnd>
                 <POS>JJ</POS>
               </token>
               <token id="2">
                 <word>discovery</word>
                 <lemma>discovery</lemma>
                 <CharacterOffsetBegin>372</CharacterOffsetBegin>
                 <CharacterOffsetEnd>381</CharacterOffsetEnd>
                 <POS>NN</POS>
                </token>
                [...]

   Here, the document logical structure information is added to the CoreNLP XML
   output by the use of the `section`, `type` and `confidence` attributes. We
   use the classification categories proposed by
   `Luong et al. (2012) <https://www.comp.nus.edu.sg/~kanmy/papers/ijdls-SectLabel.pdf>`_.
   In pke, document logical structure information is exploited by the WINGNUS
   model and the following values are handled:

   .. code::

     section="title|abstract|introduction|related work|conclusions"
     type="sectionHeader|subsectionHeader|subsubsectionHeader|bodyText"


   To read a CoreNLP XML document:

   .. code::

     extractor = pke.unsupervised.TopicRank()
     extractor.load_document(input='/path/to/input.xml')