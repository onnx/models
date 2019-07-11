import sklearn

def postprocess(embedding):
    embedding = sklearn.preprocessing.normalize(embedding).flatten()
    return embedding