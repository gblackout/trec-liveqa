#!/usr/bin/python
# -*- coding: utf-8 -*-

import glob
import sys
import re
import json

def save_json(Questions, filename):
    j = json.dumps(Questions, indent=4)
    f = open(filename, 'w')
    print >> f, j
    f.close()

first_level_fields = open('../dictionaries/fields', 'r').read().lower().split('\n')
first_level_fields = [e.strip() for e in first_level_fields]


synonym_fields = [
    ['chief complaint', 'chief complaint/reason for admission'],
    ['history', 'history of present illness', 'history of the present illness'],
    ['past medical history', 'pmh', 'medical history'],
    ['past surgical history', 'psh', 'surgical history'],
    ['brief hospital course', 'hospital course', 'hospital course: (by system)', 'hospital course (continued)',
     'concise summary of hospital course by issue/system', 'summary of hospital course by systems',
     'hospital course by system', 'summary of hospital course'],
    ['medications', 'medications prior to admission'],
    ['discharge medications', 'prescription medications', 'post discharge medications',
     'medications on discharge', 'medications at discharge'],
    ['disposition', 'discharge disposition'],
    ['discharge diagnoses', 'final diagnosis', 'final discharge diagnosis', 'final discharge diagnoses',
     'discharge diagnosis', 'interim summary diagnosis',
     'discharge (death) diagnoses'],
    ['discharge condition', 'condition on discharge', 'condition at discharge'],
    ['discharge instructions', 'general discharge instructions', 'followup instructions',
     'follow-up instructions', 'discharge instructions/followup', 'discharge plan',
     'follow up plans', 'follow up', 'follow-up', 'recommended follow up', 'assessment/plan',
     'issues and plan arising from this admission', 'discharge followup', 'additional follow-up instructions'],
    ['laboratory examination', 'laboratory', 'laboratories', 'laboratory data', 'laboratories and studies',
     'physical exam', 'physical examination'],
    ['laboratory data on admission', 'laboratories upon admission', 'laboratories on admission', 'laboratory on admission',
     'exam on admission', 'laboratory data on', 'admission laboratory data', 'admission laboratories',
     'examination on admission', 'pertinent laboratory data on presentation', 'pertinent laboratory values on presentation',
     'physical examination on presentation', 'physical exam on arrival to', 'physical exam on admission',
     'physical examination on admission'],
    ['physical examination on discharge', 'physical examination upon discharge', 'physical exam upon discharge',
     'physical exam on discharge', 'pertinent laboratory values on discharge', 'discharge pe', 'discharge exam'],
    ['admission diagnoses', 'admitting diagnosis']
]


sub_fields = {
    'pertinent results': [
        'main',
        'laboratory results',
        'admission labs',
        'discharge labs',
        'labs on admission',
        'labs on discharge',
        'labs on transfer to floor',
        'labs on admission significant for',
        'labs on transfer from icu to floor',
        'labs during hospital course',
        'laboratory studies on admission',
        'laboratory studies on discharge',
        'on presentation',
        'on discharge',
        'other results',
        'on admission',
        'microbiology',
        'other studies',
        'upon discharge',
    ],
    'laboratory examination':[
        'main',
        'on admission',
        'on discharge',
        'post surgical physical exam',
        'admission',
        'exam on discharge',
        'admission exam',
        'admission physical exam',
        'upon admission',
        'at discharge',
        'physical exam on discharge',
        'initial physical exam',
        'physical exam at time of transfer to medical floor',
        'physical exam upon transfer to micu',
        'on presentation to icu'
    ],
    'physical examination on discharge': [
        'main',
        'physical exam',
        'discharge labs',
    ],
    'discharge diagnoses': [
        'main',
        'primary diagnoses',
        'primary',
        'primary diagnosis',
        'secondary diagnoses',
        'secondary',
        'secondary diagnosis',
    ],
    'medications': [
        'main',
        'medications at home',
        'meds on admission',
        'meds on transfer',
        'on transfer from micu',
        'home',
    ]
}

def replace_midparen(str):
    str = str.strip()
    m = re.search('\[', str)
    while m:
        m2 = re.finditer('\]', str)
        found = False
        for m2m in m2:
            if m2m.start() > m.start():
                if 'name' in str[m.start():m2m.end()]:
                    tag = ' <NAME> '
                elif 'hospital' in str[m.start():m2m.end()]:
                    tag = ' <HOSPITAL> '
                else:
                    tag = ' <TIME> '
                str = str[:m.start()].strip() + tag + str[m2m.end():].strip()
                found = True
                break
        if not found: break
        m = re.search('\[', str)
    return str.strip()


def remove_paren(str):
    return remove_midparen(remove_smallparen(str)).strip()

def remove_midparen(str):
    m = re.search('\[', str)
    while m:
        m2 = re.finditer('\]', str)
        found = False
        for m2m in m2:
            if m2m.start() > m.start():
                str = str[:m.start()].strip() + ' ' + str[m2m.end():].strip()
                found = True
                break
        if not found: break
        m = re.search('\[', str)
    return str.strip()

def remove_smallparen(str):
    m = re.search('\(', str)
    while m:
        m2 = re.finditer('\)', str)
        found = False
        for m2m in m2:
            if m2m.start() > m.start():
                str = str[:m.start()].strip() + ' ' + str[m2m.end():].strip()
                found = True
                break
        if not found: break
        m = re.search('\(', str)
    return str.strip()


class field(object):
    def __init__(self, name):
        self.name = name
        self.children = None
        self.synonym = None

    def set_children(self, l):
        self.children = l

    def set_synonym(self, syn):
        self.synonym = syn

    def get_name(self):
        return self.name

    def get_synonym(self):
        return self.synonym or self.name

    def get_children(self):
        return self.children

    def is_tag(self):
        return self.get_synonym() == self.name

    def has_children(self):
        return self.children != None

field_instances = {}
all_fields_str = set()

for field_name in first_level_fields:
    field_instances[field_name] = field(field_name)
    all_fields_str.add(field_name)
for sfields in synonym_fields:
    syn = sfields[0]
    for field_name in sfields:
        assert field_name in field_instances
        field_instances[field_name].set_synonym(syn)
for key, value in sub_fields.items():
    assert key in field_instances
    field_instances[field_instances[key].get_synonym()].set_children(value)
    for v in value:
        all_fields_str.add(v)


syn_fields = [field for field in field_instances.values() if field.is_tag()]

def process_one(note, outputfile=None):
    patient = {}
    for field in syn_fields:
        if not field.has_children():
            patient[field.name] = ''
        else:
            patient[field.name] = {}
            for child_field_name in field.get_children():
                patient[field.name][child_field_name] = ''

    lines = note.lower().split('\n')
    # lines = [remove_paren(line) for line in lines]
    lines = [replace_midparen(line) for line in lines]
    tag_name, subtag = None, None
    i = 0
    while i < len(lines):
        line = lines[i]

        if i == 0:
            parts = line.split(',')
            if len(parts) < 3:
                return patient
            patient['admission id'] = int(parts[2])
        if not line or line == '':
            i += 1
            continue
        m = re.search(':', line)
        if m and line[:m.start()] in all_fields_str:
            temp_name = line[:m.start()].strip()
            if temp_name in field_instances: # its a first level field name
                tag_name = field_instances[temp_name].get_synonym()
                subtag = None
            else:
                for field_name, field in field_instances.items():
                    if field.has_children():
                        for sub_field_name in field.get_children():
                            if temp_name == sub_field_name and tag_name == field_name:
                                subtag = temp_name
                # assert subtag is not None

            lines[i] = line[m.end():].strip()
        elif tag_name is not None and subtag is not None:      # add to current tag's subfield
            patient[tag_name][subtag] += '\r' + line.replace('\n', ' ')
            i += 1
        elif tag_name is not None:
            if field_instances[tag_name].has_children():
                patient[tag_name]['main'] += '\r' + line.replace('\n', ' ')
            else:
                patient[tag_name] += '\r' + line.replace('\n', ' ')
            i += 1
        else:
            i += 1
    save_json(patient, outputfile)
    return patient


def transfer_to_json():
    # transfer text files into json format with specified fields
    count_longfile = 0
    count_nofields = 0
    count_processfiles = 0
    files = glob.glob('../notes/*.txt')
    files_toolong_fo = open('../output/files_toolong.txt', 'w')
    files_have_no_fields_fo = open('../output/files_have_no_fields.txt', 'w')
    files_have_no_fields_list = []
    assert len(files) > 0
    print('[INFO] total number of files is {0}'.format(str(len(files))))

    for file in files:
        # if 'notes/109.txt' != file: continue
        print(file)
        f = open(file, 'r').read().lower().replace('-', ' ').replace('*', ' ')
        if len(f.split()) > 10000:
            count_longfile += 1
            files_toolong_fo.write(file)
            files_toolong_fo.write('\n')
            continue
        res = process_one(f, outputfile='../notes_json_all_fields_tag/' + file.split('/')[-1].split('.txt')[0] + '.json')
        if not res:
            files_have_no_fields_fo.write(file)
            files_have_no_fields_fo.write('\n')
            count_nofields += 1
            files_have_no_fields_list.append(file)
        else:
            count_processfiles += 1

    print(files_have_no_fields_list)
    print('There are totally {0} long files'.format(str(count_longfile)))
    print('There are totally {0} files have no field'.format(str(count_nofields)))
    print('count_processfiles is ' + str(count_processfiles) )
    print('total files is ' + str(len(files)))
    assert count_longfile + count_nofields + count_processfiles == len(files)

if __name__ == "__main__":
    transfer_to_json()