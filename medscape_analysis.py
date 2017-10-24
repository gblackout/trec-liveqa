import json
import re


synonyms = {'0': ['Treatment;Approach Considerations',
                          'Overview;Practice Essentials;Management',
                          'Overview;Treatment & Management',
                          'Overview;Practice Essentials;Diagnosis and management',
                          'Medication;Medication Summary',
                          'Treatment;Medical Care',
                          'Treatment;Surgical Care',
                          'Treatment;Medical Therapy',
                          'Treatment;Surgical Therapy',
                          'Treatment;Pharmacologic Therapy',
                          'Treatment;Surgical Intervention',
                          'Treatment;Antibiotic Therapy',
                          'Treatment;Other Treatment',
                          'Treatment;Medical Issues/Complications',
                          'Treatment;Complications',
                          'Treatment;Diet and Activity',
                          'Treatment;Activity',
                          'Treatment;Diet',
                          'Treatment;Long-Term Monitoring',
                          'Treatment;Consultations',
                          'Overview;Indications',
                          'Overview;Contraindications',
                          'Overview;Treatment Protocols',
                          'Treatment;Prevention',
                          'Overview;Complication Prevention',
                          'Overview;Medications',
                          'Overview;Prevention',
                          'Overview;Management',
                          'Overview;Indications and Contraindications',
                          'Overview;Therapy'],
            '1': ['Overview;Practice Essentials;Practice Essentials',
                            'Overview;Overview',
                            'Overview;Background',
                            'Overview;Problem',
                            'Overview;Epidemiology',
                            'Treatment;Approach Considerations',
                            'Medication;Medication Summary',
                            'Overview;Patient Education',
                            'Overview;Definition',
                            'Overview;Epidemiology',
                            'Overview;Etiology',
                            'Overview;Prognosis and Predictive Factors'],
            '2': ['Overview;Mortality/Morbidity',
                               'Overview;Epidemiology',
                               'Overview;Frequency',
                               'Overview;Epidemiology and Prognosis'],
            '3': ['Overview;Prognosis',
                          'Overview:Outcomes',
                          'Overview;Mortality/Morbidity',
                          'Overview;Prognosis and Predictive Factors',
                          'Overview;Epidemiology and Prognosis',
                          'Treatment;Outcome and Prognosis',
                          'Overview;Background'],
            '4': ['Overview;Practice Essentials;Signs and symptoms',
                        'Overview;Presentation;Symptoms',
                        'Overview;Presentation',
                        'Presentation;History',
                        'Overview;Practice Essentials;Complications',
                        'Overview;Complications',
                        'Overview:Outcomes',
                        'Overview;Problem',
                        'Overview;Clinical Presentation'],
            '5': ['Overview;Practice Essentials;Diagnosis',
                          'Overview;Practice Essentials;Diagnosis and management',
                          'Overview;Diagnosis',
                          'Workup;Approach Considerations',
                          'Workup;Laboratory Studies',
                          'Workup;Lab Studies',
                          'Workup;Diagnostic Procedures',
                          'Overview;Interpretation',
                          'Presentation;Physical',
                          'Presentation;Physical Examination',
                          'Presentation;History and Physical Examination',
                          'Workup;Other Tests',
                          'Workup;Histologic Findings',
                          'Overview;Differential Diagnosis',
                          'Presentation;Complications'],
            '6': ['Overview;Etiology',
                      'Overview;Pathophysiology and Etiology',
                      'Presentation;Causes',
                      'Overview;Pathophysiology'],
            '7': ['Placeholder'],
            '8': ['Placeholder'],
            '9': ['Placeholder']
            }


def foo():
    content_ls = json.load(open('../crawl-data/disease.json'))['data']
    sup_ls = json.load(open('../crawl-data/disease_fix.json'))['data']

    content_ls += sup_ls
    res = {}

    url_reg = re.compile(r'article/(\d+)-(.*)$')

    cnt_diff = 0

    for content in content_ls:
        m = url_reg.search(content['url'])

        if m is None:
            continue

        page_id = m.group(1)

        if page_id not in res:
            res[page_id] = {'content':{}, 'title':{}, 'url':{}, 'type':{}}
            tab_name = content['content'].keys()[0]

            if tab_name == '':
                new_tab_name = 'Overview'
                res[page_id]['content'][new_tab_name] = content['content'][tab_name]
                res[page_id]['title'] = content['name']
                res[page_id]['url'] = content['url']
                res[page_id]['type'] = content['type']
                continue

            res[page_id]['content'][tab_name] = content['content'][tab_name]
            res[page_id]['title'] = content['name']
            res[page_id]['url'] = content['url']
            res[page_id]['type'] = content['type']

        else:
            tab_name = content['content'].keys()[0]

            if tab_name in res[page_id]['content']:
                assert tab_name == 'Overview', tab_name
                assert res[page_id]['content'][tab_name] == content['content'][tab_name], page_id
                cnt_diff += 1
                continue

            res[page_id]['content'][tab_name] = content['content'][tab_name]

            # overview url title and type get highest priority
            if tab_name == 'Overview':
                res[page_id]['title'] = content['name']
                res[page_id]['url'] = content['url']
                res[page_id]['type'] = content['type']

    print '!!!!!!!!', cnt_diff

    json.dump(res, open('../crawl-data/group_disease.json', 'w'))

from copy import deepcopy
camel_reg = re.compile(r'([a-z])([A-Z])')


def xxx(e):
    if type(e) != dict:
        raise ValueError

    newe = deepcopy(e)
    for k in e:
        if type(e[k]) == unicode:
            newe[k] = camel_reg.sub(r'\1 \2', e[k])
        else:
            newe[k] = xxx(e[k])

    return newe



def split_wd():
    jn = json.load(open('../crawl-data/group_disease.json'))

    ks = jn.keys()

    for id in ks:
        content = jn[id]['content']
        jn[id]['content'] = xxx(content)

    json.dump(jn, open('../crawl-data/group_disease_split.json', 'w'))



def jndata():
    import pickle
    # visit all json files, extract X and Y

    res = []
    file_labels = pickle.load(open('type_data'))
    for i, (text_one, label_vec) in enumerate(file_labels):
        types = [i for i in xrange(len(label_vec)) if label_vec[i] > 0]
        res.append({'text':text_one, 'types':types})

    json.dump(res, open('jndata.json', 'w'))


if __name__ == '__main__':
    # foo()
    # split_wd()
    json.dump(synonyms, open('../crawl-data/synonyms.json', 'w'))
    # jndata()