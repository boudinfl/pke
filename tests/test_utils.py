#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pke


def test_load_reference():
    """Various tests for loading a reference file."""

    id = 'C-41'

    g1 = pke.utils.load_references(input_file='tests/data/reference.json',
                                   normalize_reference=True,
                                   language="en",
                                   encoding='utf-8')

    g2 = pke.utils.load_references(input_file='tests/data/reference.stem.json',
                                   normalize_reference=False,
                                   language="en",
                                   encoding='utf-8')

    g3 = pke.utils.load_references(input_file='tests/data/reference.final',
                                   normalize_reference=True,
                                   language="en",
                                   encoding='utf-8')

    g4 = pke.utils.load_references(input_file='tests/data/reference.stem.final',
                                   normalize_reference=False,
                                   language="en",
                                   encoding='utf-8')

    assert set(g1[id]) == set(g2[id]) == set(g3[id]) == set(g4[id])


if __name__ == '__main__':
    test_load_reference()
