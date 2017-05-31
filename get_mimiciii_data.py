import numpy as np
import os
import json


def import_json(filename):
    with open(filename) as data_file:
        data = json.load(data_file)
    return data


fields = ['chief complaint', 'history', 'past medical history', 'past surgical history',
          'past surgical history', 'social history', 'family history', 'brief hospital course',
          'major surgical or invasive procedure', 'review of systems', 'pertinent results',
          'service', 'allergies', 'laboratory examination', 'laboratory data on',
          'medications on transfer', 'medications', 'medications at home', 'medications on admission']

sub_fields = {
    'pertinent results': [
        'main',
        'laboratory results',
        'admission labs',
        'labs on admission',
        'labs on admission significant for',
        'laboratory studies on admission',
        'on presentation',
        'on admission',
    ],
    'laboratory examination': [
        'main',
        'on admission',
        'admission',
        'admission exam',
        'admission physical exam',
        'upon admission',
        'initial physical exam',
        'on presentation to icu'
    ],
    'medications': [
        'main',
        'medications at home',
        'meds on admission',
        'home',
    ]
}


def get_admission_text(patient):
    # return a sentence containing content of all fields of a patient
    res = []
    for field in patient:
        if field in fields:
            if field not in sub_fields:
                res.append(patient[field].replace('\n', '\r'))
            else:
                for subfield in sub_fields[field]:
                    res.append(patient[field][subfield].replace('\n', '\r'))
    res = ' '.join(res)
    res = ' '.join(res.split())  # combine multiple spaces into one space
    return res


def get_admission_text_from_json(file):
    return get_admission_text(import_json(file))
