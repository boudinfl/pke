from __future__ import absolute_import
import os
import sys
sys.path.append('pke')
print(sys.path)
from unsupervised.statistical.rake import RAKE

def test_rake_candidate_selection():
    textData = [
        "MATLAB provides tools for scientists and engineers. MATLAB is used by scientists and engineers."
        "Analyze text and images. You can import text and images."
        "Analyze text and images. Analyze text, images, and videos in MATLAB."]
    extractor = RAKE()
    extractor.load_document(input=textData)
    extractor.generate_candidate_keywords()
    print(extractor.candidates)


if __name__ == '__main__':
    test_rake_candidate_selection()