import os
import pickle
import pprint
import glob
from os.path import join as joinpath
from copy import deepcopy


def printdoc(doc):
    """
    pretty print a dictionary-format of a admission document
    
    input
    -----
    doc: dict generated by generate_data.main
    """

    doc = deepcopy(doc)

    def str2list(doc):
        for k, v in doc.iteritems():
            if type(v) == type(u''):

                if len(v) < 1:
                    continue
                elif v[:1] == u'\n':
                    v = v[1:]

                if u'\n' in v:
                    doc[k] = v.split(u'\n')
                else:
                    doc[k] = v

            elif type(v) == dict:
                doc[k] = str2list(v)

        return doc

    pprint.pprint(str2list(doc))

    print '\n', '=' * 100, '\n'


def foo():
    data_path = '../data/data_as_dict'
    filenames = glob.glob(joinpath(data_path, '*'))

    bound = 5

    for i, filename in enumerate(filenames):
        doc = pickle.load(open(filename))
        printdoc(doc)

        if i == bound:
            break


def bar():
    data_path = '../data/notes_json_all_fields'
    filenames = glob.glob(joinpath(data_path, '*'))
    from generate_data import import_json

    bound = 5

    for i, filename in enumerate(filenames):
        doc = import_json(filename)
        pprint.pprint(doc)
        print '\n', '=' * 100, '\n'
        if i == bound:
            break


if __name__ == '__main__':
    bar()
    # foo()
