import glob
import numpy as np
from get_medication import is_number_index, import_json

def rm_punc(line):
    # remove punctuations at the end of a line
    line = line.strip()
    if line.isdigit():
        return ''
    while line and line[-1] in ['.', '#', '%', ',', ':']:
        line = line[:-1].strip()
    while line and line[0] in ['.', '#', '%', ',', ':', ')']:
        line = line[1:].strip()
    return line

abbrs = ['vs', 'mr', 'dr', 'ms', 'p.o', 'm.d']

def type(content, type):
    content = content.strip()
    for abbr in abbrs:
        content = content.replace(abbr+'. ', '&&&' + abbr + '&&&')
    if type == 1:
        sents = type1(content)
    elif type == 2:
        sents = type2(content)
    elif type == 3:
        sents = type3(content)
    for i in range(len(sents)):
        for abbr in abbrs:
            sents[i] = sents[i].replace('&&&' + abbr + '&&&', abbr+'. ')
    sents = [rm_punc(sent) for sent in sents if rm_punc(sent)]
    return sents

def type1(content):
    # type1: replace \r to space, and seperate by .
    content = ' '.join(content.split())
    sents = content.split('. ')
    return sents

def type2(content):
    # type2: seperated by \r
    sents = content.split('\r')
    return sents

def type3(content):
    # type3: for line seperated or numeric seperated (mainly medications)
    sents = []
    if is_number_index(content):
        lines = content.split('\r')
        digit = 1
        for line in lines:
            line = line.strip()
            if line[:3] == str(digit) + '. ':
                sents.append(line[3:])
            elif sents:
                sents[-1] = sents[-1] + ' ' + line
    # else:
    #     sents = content.split('\r')
    return sents


files = glob.glob('../notes_json_all_fields/*.json')
num_files = len(files)
print('number of files is {}'.format(num_files))
# random_indices = np.random.permutation(num_files)
notes = {}

count = 0
fo = open('../ner_output/history', 'w')
for i in range(num_files):
    file = files[i]
    print(file)
    patient = import_json(file)
    if 'history' in patient:
        c = type(patient['history'], type=1)
        if c:
            fo.write('####### {}\n'.format(file))
            fo.write('\n'.join(c) + '\n\n')
            count += 1

print('selected files number is {}'.format(count))
fo.close()
