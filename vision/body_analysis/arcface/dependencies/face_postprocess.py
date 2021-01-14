# SPDX-License-Identifier: Apache-2.0

import sklearn

def postprocess(embedding):
    embedding = sklearn.preprocessing.normalize(embedding).flatten()
    return embedding
