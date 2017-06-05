from copy import deepcopy
from generate_data import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


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


def hypertension_drugs(uni_con_path):

    # med_dict = {}
    # med_reg = re.compile(r'\w+')
    # type_cnt = 0
    # with open('../hypertension_med_list') as f:
    #     for line in f:
    #         generic_name, brand_names = line.split(':')
    #         brand_names = [e.strip().lower() for e in brand_names.split(',') if med_reg.search(e)]
    #         med_dict.update([(e, [type_cnt, 0, 0]) for e in brand_names + [generic_name.lower()]])
    #         type_cnt += 1
    #
    # pprint.pprint(med_dict)
    #
    # for i, filename in enumerate(allfilenames(uni_con_path)):
    #     print '%i\t / \t%i' % (i, 52682), filename
    #
    #     nc = NoteContainer(filename, mode=1)
    #     admed_str = ' '.join(nc.admission_medications).lower() if nc.admission_medications else ''
    #     dismed_str = ' '.join(nc.discharge_medications).lower() if nc.discharge_medications else ''
    #
    #     for k in med_dict:
    #         if k in admed_str:
    #             med_dict[k][1] += 1
    #         if k in dismed_str:
    #             med_dict[k][2] += 1
    #
    # pickle.dump(med_dict, open('med_dict', 'w'))

    med_dict = pickle.load(open('med_dict'))
    med_list = sorted([(k, v) for k, v in med_dict.iteritems()], key=lambda x: x[1][0])
    x_ticks, allcnts = zip(*med_list)
    type_list, adm_y, dis_y = zip(*allcnts)
    x = range(len(x_ticks))

    print 1

    fig, ax1 = plt.subplots(figsize=(30, 10))

    ax1.plot(x, adm_y, 'b-')
    ax1.set_xlabel('medications')
    plt.xticks(x, x_ticks, rotation=50, ha='right')
    plt.setp(ax1.get_xticklabels(), fontsize=10)

    print 1.5
    colormap = plt.cm.gist_ncar

    import random

    colors = [colormap(i) for i in np.linspace(0, 0.9, len(set(type_list)))]
    random.shuffle(colors)
    cnt = 0
    for xtick in ax1.get_xticklabels():
        xtick.set_color(colors[med_dict[x_ticks[cnt]][0]])
        cnt += 1

    print 2

    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('admission frequency', color='b')
    ax1.tick_params('y', colors='b')
    ax1.xaxis.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(x, dis_y, 'r-')
    ax2.set_ylabel('discharge frequency', color='r')
    ax2.tick_params('y', colors='r')

    print 3

    fig.tight_layout()
    # plt.show()
    plt.savefig('123.png')

    print 4





if __name__ == '__main__':
    hypertension_drugs('uni_containers_tmp/')





























