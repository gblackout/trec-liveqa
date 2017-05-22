import glob
import sys
import re
sys.path.append('../../../Common/')
from data_common import save_json, import_json
from collections import Counter



fields = ['allergies', 'chief complaint', 'history of present illness', 'past medical history', 'social history',
            'family history', 'initial exam', 'medications on admission', 'discharge diagnosis', 'secondary',
          'initial physical exam', 'diagnoses', 'medical history', 'past surgical history', 'pmh', 'psh']
fields = [field.lower() for field in fields]

all_fields = fields + ['followup instructions', 'discharge instructions', 'discharge condition', 'general', 'physical exam',
                       'attending', 'major surgical or invasive procedure', 'neurological examination', 'pertinent results',
                       'brief hospital course', 'discharge medications', 'discharge disposition', 'discharge condition',
                       'discharge instructions', 'followup instructions', 'date of birth', 'service'
                       ]
all_fields = [field.lower() for field in all_fields]

def has_all_fields(patient):
    for field in all_fields:
        if field not in patient:
            return False
    return True


def process_one(content, outputfile=None):
    # input is a string format content
    # output a json with specified fields
    patient = {}
    find_field = False
    for field in all_fields:
        patient[field] = ''
    tag = None
    lines = content.split('\n')
    for line in lines:
        if not line: continue
        for field in all_fields:
            if (field + ':') in line.lower():
                if field in all_fields:
                    if field == 'initial physical exam':
                        tag = 'initial exam'
                    elif field == 'diagnoses':
                        tag = 'discharge diagnosis'
                    elif field == 'medical history' or field == 'pmh':
                        tag = 'past medical history'
                    elif field == 'psh':
                        tag = 'past social history'
                    else:
                        tag = field
                    break
                else:
                    tag = None

        if tag:
            find_field = True
            patient[tag] += line.replace((field + ':'), '')

        # if has_all_fields(patient):
        #     break

    if find_field and outputfile:
        save_json(patient, outputfile)
    else:
        return None

    return patient

def transfer_to_json():
    # transfer text files into json format with specified fields
    count_longfile = 0
    count_nofields = 0
    count_processfiles = 0
    files = glob.glob('notes/*.txt')
    files_toolong_fo = open('output/files_toolong.txt', 'w')
    files_have_no_fields_fo = open('output/files_have_no_fields.txt', 'w')
    assert len(files) > 0
    print('[INFO] total number of files is {0}'.format(str(len(files))))

    for file in files:
        f = open(file, 'r').read().lower()
        if len(f.split()) > 10000:
            count_longfile += 1
            files_toolong_fo.write(file)
            files_toolong_fo.write('\n')
            continue
        res = process_one(f, outputfile='notes_json/' + file.split('/')[-1].split('.txt')[0] + '.json')
        if res is None:
            files_have_no_fields_fo.write(file)
            files_have_no_fields_fo.write('\n')
            count_nofields += 1
        else:
            count_processfiles += 1

    print('There are totally {0} long files'.format(str(count_longfile)))
    print('There are totally {0} files have no field'.format(str(count_nofields)))
    assert count_longfile + count_nofields + count_processfiles == len(files)

if __name__ == "__main__":
    transfer_to_json()