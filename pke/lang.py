# -*- coding: utf-8 -*-

"""Language resources of pke.

Lists of stopwords in different languages.

These lists were obtained from nltk's stopwords corpus data.
https://github.com/nltk/nltk_data/blob/gh-pages/packages/corpora/stopwords.zip

Langcodes.

"""

stopwords = {
    "en": ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "you\'re", "you\'ve", "you\'ll",
           "you\'d", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "she\'s", "her",
           "hers", "herself", "it", "it\'s", "its", "itself", "they", "them", "their", "theirs", "themselves", "what",
           "which", "who", "whom", "this", "that", "that\'ll", "these", "those", "am", "is", "are", "was", "were", "be",
           "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and",
           "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against",
           "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
           "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
           "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
           "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
           "don\'t", "should", "should\'ve", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "aren\'t",
           "couldn", "couldn\'t", "didn", "didn\'t", "doesn", "doesn\'t", "hadn", "hadn\'t", "hasn", "hasn\'t", "haven",
           "haven\'t", "isn", "isn\'t", "ma", "mightn", "mightn\'t", "mustn", "mustn\'t", "needn", "needn\'t", "shan",
           "shan\'t", "shouldn", "shouldn\'t", "wasn", "wasn\'t", "weren", "weren\'t", "won", "won\'t", "wouldn",
           "wouldn\'t"],
    "fr": ['au', 'aux', 'avec', 'ce', 'ces', 'dans', 'de', 'des', 'du', 'elle', 'en', 'et', 'eux', 'il', 'ils', 'je',
           'la', 'le', 'les', 'leur', 'lui', 'ma', 'mais', 'me', 'même', 'mes', 'moi', 'mon', 'ne', 'nos', 'notre',
           'nous', 'on', 'ou', 'par', 'pas', 'pour', 'qu', 'que', 'qui', 'sa', 'se', 'ses', 'son', 'sur', 'ta', 'te',
           'tes', 'toi', 'ton', 'tu', 'un', 'une', 'vos', 'votre', 'vous', 'c', 'd', 'j', 'l', 'à', 'm', 'n', 's', 't',
           'y', 'été', 'étée', 'étées', 'étés', 'étant', 'étante', 'étants', 'étantes', 'suis', 'es', 'est', 'sommes',
           'êtes', 'sont', 'serai', 'seras', 'sera', 'serons', 'serez', 'seront', 'serais', 'serait', 'serions',
           'seriez', 'seraient', 'étais', 'était', 'étions', 'étiez', 'étaient', 'fus', 'fut', 'fûmes', 'fûtes',
           'furent', 'sois', 'soit', 'soyons', 'soyez', 'soient', 'fusse', 'fusses', 'fût', 'fussions', 'fussiez',
           'fussent', 'ayant', 'ayante', 'ayantes', 'ayants', 'eu', 'eue', 'eues', 'eus', 'ai', 'as', 'avons', 'avez',
           'ont', 'aurai', 'auras', 'aura', 'aurons', 'aurez', 'auront', 'aurais', 'aurait', 'aurions', 'auriez',
           'auraient', 'avais', 'avait', 'avions', 'aviez', 'avaient', 'eut', 'eûmes', 'eûtes', 'eurent', 'aie', 'aies',
           'ait', 'ayons', 'ayez', 'aient', 'eusse', 'eusses', 'eût', 'eussions', 'eussiez', 'eussent'],
    "ge": ['aber', 'alle', 'allem', 'allen', 'aller', 'alles', 'als', 'also', 'am', 'an', 'ander', 'andere', 'anderem',
           'anderen', 'anderer', 'anderes', 'anderm', 'andern', 'anderr', 'anders', 'auch', 'auf', 'aus', 'bei', 'bin',
           'bis', 'bist', 'da', 'damit', 'dann', 'der', 'den', 'des', 'dem', 'die', 'das', 'dass', 'daß', 'derselbe',
           'derselben', 'denselben', 'desselben', 'demselben', 'dieselbe', 'dieselben', 'dasselbe', 'dazu', 'dein',
           'deine', 'deinem', 'deinen', 'deiner', 'deines', 'denn', 'derer', 'dessen', 'dich', 'dir', 'du', 'dies',
           'diese', 'diesem', 'diesen', 'dieser', 'dieses', 'doch', 'dort', 'durch', 'ein', 'eine', 'einem', 'einen',
           'einer', 'eines', 'einig', 'einige', 'einigem', 'einigen', 'einiger', 'einiges', 'einmal', 'er', 'ihn',
           'ihm', 'es', 'etwas', 'euer', 'eure', 'eurem', 'euren', 'eurer', 'eures', 'für', 'gegen', 'gewesen', 'hab',
           'habe', 'haben', 'hat', 'hatte', 'hatten', 'hier', 'hin', 'hinter', 'ich', 'mich', 'mir', 'ihr', 'ihre',
           'ihrem', 'ihren', 'ihrer', 'ihres', 'euch', 'im', 'in', 'indem', 'ins', 'ist', 'jede', 'jedem', 'jeden',
           'jeder', 'jedes', 'jene', 'jenem', 'jenen', 'jener', 'jenes', 'jetzt', 'kann', 'kein', 'keine', 'keinem',
           'keinen', 'keiner', 'keines', 'können', 'könnte', 'machen', 'man', 'manche', 'manchem', 'manchen', 'mancher',
           'manches', 'mein', 'meine', 'meinem', 'meinen', 'meiner', 'meines', 'mit', 'muss', 'musste', 'nach', 'nicht',
           'nichts', 'noch', 'nun', 'nur', 'ob', 'oder', 'ohne', 'sehr', 'sein', 'seine', 'seinem', 'seinen', 'seiner',
           'seines', 'selbst', 'sich', 'sie', 'ihnen', 'sind', 'so', 'solche', 'solchem', 'solchen', 'solcher',
           'solches', 'soll', 'sollte', 'sondern', 'sonst', 'über', 'um', 'und', 'uns', 'unsere', 'unserem', 'unseren',
           'unser', 'unseres', 'unter', 'viel', 'vom', 'von', 'vor', 'während', 'war', 'waren', 'warst', 'was', 'weg',
           'weil', 'weiter', 'welche', 'welchem', 'welchen', 'welcher', 'welches', 'wenn', 'werde', 'werden', 'wie',
           'wieder', 'will', 'wir', 'wird', 'wirst', 'wo', 'wollen', 'wollte', 'würde', 'würden', 'zu', 'zum', 'zur',
           'zwar', 'zwischen']
}

langcodes = {
       "ar": "arabic",
       "da": "danish",
       "du": "dutch",
       "en": "english",
       "fi": "finnish",
       "fr": "french",
       "ge": "german",
       "hu": "hungarian",
       "it": "italian",
       "no": "norwegian",
       "po": "portuguese",
       "ro": "romanian",
       "ru": "russian",
       "sp": "spanish",
       "sw": "swedish",
       "ja": "japanese"
}
