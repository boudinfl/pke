# -*- coding: utf-8 -*-
# Python Keyphrase Extraction toolkit: unsupervised models

from __future__ import absolute_import

from pke.supervised.api import SupervisedLoadFile
from pke.supervised.feature_based.kea import Kea
from pke.supervised.feature_based.topiccorank import TopicCoRank
from pke.supervised.feature_based.wingnus import WINGNUS
from pke.supervised.neural_based.seq2seq import Seq2Seq
