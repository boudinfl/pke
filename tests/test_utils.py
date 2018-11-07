#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pke


def test_load_reference_in_json_format():
    """Various tests for loading a reference file in json format"""

    gold1 = pke.utils.load_references(input_file='tests/data/reference.json',
                                      normalize_reference=True,
                                      language="en",
                                      encoding='utf-8')

    gold2 = pke.utils.load_references(input_file='tests/data/reference.stem.json',
                                      normalize_reference=False,
                                      language="en",
                                      encoding='utf-8')

    assert set(gold1['C-41']).issubset(set(gold2['C-41']))





if __name__ == '__main__':
    test_load_reference_in_json_format()
