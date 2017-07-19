import json
import re


synonyms = {'treatment': ['',],
            'information': ['Background',],
            'disease_symptom': ['', ],
            'susceptibility': ['Epidemiology', ],
            'prognosis': ['', ],
            'diagnosis': ['', ],
            'cause': ['', ],
            'organization': ['', ],
            'disease_prevent': ['', ],
            'drug_info': ['', ],
            'disease_intertaction': ['', ],
            'drug_interaction': ['', ],
            }


def foo():
    content_ls = json.load(open('../crawl-data/disease.json'))['data']
    res = {}

    url_reg = re.compile(r'article/(\d+)-(.*)$')

    cnt = 0
    cnt_diff = 0

    for content in content_ls:
        m = url_reg.search(content['url'])

        if m is None:
            continue

        page_id = m.group(1)

        if page_id not in res:
            res[page_id] = {}
            tab_name = content['content'].keys()[0]

            if tab_name == '':
                new_tab_name = 'Overview'
                res[page_id][new_tab_name] = content['content'][tab_name]
                continue

            res[page_id][tab_name] = content['content'][tab_name]

        else:
            tab_name = content['content'].keys()[0]

            if tab_name in res[page_id]:

                if m.group(2) != 'differentia':
                    cnt += 1
                    print m.group(2)+' '+ page_id
                    continue

                assert res[page_id][tab_name] == content['content'][tab_name], page_id
                cnt_diff += 1

            assert tab_name != ''

            res[page_id][tab_name] = content['content'][tab_name]

    print '!!!!!!!!', cnt, cnt_diff

    json.dump(res, open('../crawl-data/group_disease.json', 'w'))


def bar():

    import json
    with open('filteredQuestion.json') as k:
        jn = json.load(k)
        X = []
        for i, entry in enumerate(jn):
            text_one = entry['title'] + '. ' + entry['content']
            single_x, _ = preprocess_mimiciii.DataLoader.parse_single(text_one, joinpath(model_path, 'mat'), 'stpwd')
            X.append(single_x)
        X = np.concatenate(X, axis=0)



if __name__ == '__main__':
    foo()