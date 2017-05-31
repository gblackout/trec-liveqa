import glob
import json
from os.path import join as joinpath
import os
import shutil
import pickle
import warnings
import traceback
import pprint
import re
from collections import Counter

id_template = ["ROW_ID",
               "SUBJECT_ID",
               "HADM_ID",
               "CHARTDATE",
               "CHARTTIME",
               "STORETIME",
               "CATEGORY",
               "DESCRIPTION",
               "CGID",
               "ISERROR",
               "TEXT"]

synonym = {'allergy': ['allergies'],

           'chief_complaint': ['chief complaint'],

           'historyOf_present_illness': ['history of present illness',
                                         'history of the present illness'],

           'past_medical_history': ['past medical history',
                                    'major surgical or invasive procedure',
                                    'past surgical history'],

           'social_history': ['social history'],

           'family_history': ['family history'],

           'initial_exam': ['physical exam',
                            'pertinent results',
                            'impression',
                            'physical examination',
                            'findings',
                            'admission labs',
                            'physical examination on presentation'],

           'admission_medications': ['medications on admission'],

           'discharge_medications': ['discharge medications',
                                     'medications on discharge']
           }

# make it more efficient for look-up
heading_lookup = {}
for k, syns in synonym.iteritems():
    heading_lookup.update([(syn, k) for syn in syns])

todo = ['medications',
        'history',
        'on admission',
        'admission',
        'admission exam',
        'admission medications',
        'physical examination on admission',
        'pe',
        'admission physical exam',
        'on discharge',
        'upon discharge',
        'at discharge',
        'discharge',
        'pmh', 'psh', 'pmhx', 'pshx', 'hpi', 'family hx',
        'indication',
        'medical condition',
        'medications on transfer',
        'medication changes',
        'medications at home',
        'addendum',
        'home medications',
        'the following changes were made to your medications',
        'we made the following changes to your medicines',
        'we made the following changes to your medications',
        'we have made the following changes to your medications',
        'the following changes have been made to your medications',
        'meds on transfer',
        'medications prior to admission',
        'meds',
        'admission diagnosis',
        'admission physical examination',
        'clinical history',
        'other past medical history',
        'home meds']


class NoteContainer:
    def __init__(self, arg, mode=0):

        self.extnotes = None

        # raw_heading
        self.raw_headings = None

        # fields suggested by Carol
        self.allergy = None
        self.chief_complaint = None
        self.historyOf_present_illness = None
        self.past_medical_history = None
        self.social_history = None
        self.family_history = None
        self.initial_exam = None
        self.admission_medications = None
        self.discharge_medications = None

        # attributes that are interesting
        self.age = None

        if mode == 0:
            self.note = self.preprop(arg)
            self.ID_fields = self.extract_IDfields()
        else:
            self.load(arg)

    def preprop(self, seq):
        # remove [**<anything>**] placeholder in text
        seq = re.sub(r'\[\*\*[^\[]*\*\*\]', '', seq)

        # remove \r
        seq = re.sub(r'\r\n', '\n', seq)

        return seq

    def extract_IDfields(self):

        candidates = re.findall(r'([0-9 ]*),'
                                r'([0-9 ]+),'
                                r'([0-9 ]*),'
                                r'([a-zA-Z0-9 \":\-]*),'
                                r'([a-zA-Z0-9 \":\-]*),'
                                r'([a-zA-Z0-9 \":\-]*),'
                                r'([a-zA-Z0-9 \":\-]*),'
                                r'([a-zA-Z0-9 \":\-]*),'
                                r'([a-zA-Z0-9 \":\-]*),'
                                r'([a-zA-Z0-9 \":\-]*),'
                                r'([a-zA-Z0-9 \":\-/\.]*)$', self.note, flags=re.MULTILINE | re.UNICODE)

        # no field is found
        if not candidates:
            raise ValueError('cannot find ID in file')

        # too many fields are found, may happen in nursing report
        if len(candidates) > 1:
            warnings.warn('multiple ID row found in file')

        # produce a list of dicts with keys in id_template and values in cand
        return [dict((id_template[i], cand[i]) for i in xrange(len(id_template))) for cand in candidates]

    def get_notetype(self):
        return self.ID_fields[0]['CATEGORY'], self.ID_fields[0]['DESCRIPTION']

    def get_subID(self):
        return self.ID_fields[0]['SUBJECT_ID']

    def get_admID(self):
        return self.ID_fields[0]['HADM_ID']

    def save(self, _path):
        pickle.dump(self.__dict__, open(_path, 'w'))

    def load(self, _path):
        for k in self.__dict__:
            self.__dict__[k] = None

        attr_dict = pickle.load(open(_path))
        for k, v in attr_dict.iteritems():
            self.__dict__[k] = v

    def extract_headings(self):
        self.raw_headings = set()
        isfound = False

        for rawtext in [self.note] + (self.extnotes if self.extnotes else []):

            m = re.findall(r'(^[a-zA-Z ]*):[^:]*$', rawtext, flags=re.MULTILINE | re.UNICODE)

            if not m:
                continue

            self.raw_headings.update([e.strip().lower() for e in m])
            isfound = True

        if not isfound:
            raise RuntimeError('no heading found in %s' % self.get_admID())

    def extract_contents(self):

        # assuming content starts right after : and ends with an empty line
        blank_reg = re.compile(r'^\s*$', flags=re.MULTILINE)
        heading_reg = re.compile(r'(^[a-zA-Z ]*):[^:]*$', flags=re.MULTILINE | re.UNICODE)

        for rawtext in [self.note] + (self.extnotes if self.extnotes else []):

            matches = heading_reg.finditer(rawtext) # return an iterator
            matches = [m for m in matches] # we want a list

            # filter out headings of no interest
            filtered_matches = []
            for i, m in enumerate(matches):
                if not m:
                    continue

                heading = rawtext[m.start(1):m.end(1)].strip().lower() # note it's 1 not 0

                if heading not in heading_lookup:
                    continue

                filtered_matches.append(m)

            numOf_matches = len(filtered_matches)
            for i, m in enumerate(filtered_matches):
                heading = rawtext[m.start(1):m.end(1)].strip().lower() # note it's 1 not 0
                field_name = heading_lookup[heading]

                # assuming the index starting right after : of a matching heading
                content_start_ind = m.end(1)+1
                # assuming the index ending right before the next match of a heading (or end of string)
                content_end_ind = len(rawtext) if i + 1 >= numOf_matches else filtered_matches[i + 1].start(0)

                blank_match = blank_reg.search(rawtext, pos=content_start_ind, endpos=content_end_ind)

                # if match then append text to the class attribute, note that we use list since there can be multiple
                if blank_match:
                    content = rawtext[content_start_ind:blank_match.start(0)]
                    if self.__dict__[field_name]: # is not None
                        self.__dict__[field_name].append(content)
                    else: # is None
                        self.__dict__[field_name] = [content]

    def extract_age(self):
        age_reg = re.compile(r'(\d{1,3})[ \-]?(y/?o|years?[ \-]old)', flags=re.IGNORECASE)

        # do a greedy search
        for rawtext in [self.note] + (self.extnotes if self.extnotes else []):
            m = age_reg.search(rawtext)
            if m:
                self.age = m.group(1)
                return

        # # TODO
        # if self.get_admID() in ['190258', '194747', '143736', '179610', '120156', '180938', '108998']:
        #     self.age = '10'
        #     return

        raise RuntimeError('Age content not found in %s\ntext:\n%s' % (self.get_admID(), self.note))

    def fields_asText(self):
        """
        concatenate all fields of interest and return a single string (excluding discharge), 
        used as input of text_cnn model. All punctuations are removed, text is lowered
        
        output
        -----
        
        string containing all fields of interest
        """

        punc_reg = re.compile(r'[^a-z0-9 ]', flags=re.IGNORECASE)
        space_reg = re.compile(r'\s+')

        res = []

        for k in synonym:

            if k == 'discharge_medications':
                continue

            concate_text = ''.join(self.__dict__[k])
            concate_text = punc_reg.sub(' ', concate_text)
            concate_text = space_reg.sub(' ', concate_text)

            res.append(concate_text)

        return ' '.join(res).lower()


def import_json(filename):
    with open(filename) as data_file:
        data = json.load(data_file)
    return data


def allfilenames(_dir):
    return glob.glob(joinpath(_dir, '*'))


def extract_fields(json_obj):
    """
    extract structured text from json file 
    
    
    input
    -----
    json_obj: the json object
    
    
    output
    -----
    {
    field_name:string,
    ...
    field_name:{
                subfield_name:string, 
                subfield_name:string, 
                ...
                }, 
    ...
    }
    
    """

    res = {}
    from get_mimiciii_data import fields, sub_fields

    for field in json_obj:
        # skip if not valid field
        if field not in fields:
            continue

        # if field not contains sub-fields
        if field not in sub_fields:
            res[field] = json_obj[field].replace('\r', '\n')

        else:
            res[field] = {}
            for subfield in sub_fields[field]:
                res[field][subfield] = json_obj[field][subfield].replace('\r', '\n')

    return res


def main(json_filepath, output_path):
    from get_medication import get_med, is_number_index

    filenames = glob.glob(joinpath(json_filepath, '*.json'))
    numOf_file = len(filenames)
    assert numOf_file > 0

    # clear old dir content
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)

    os.mkdir(output_path)

    # for each json file
    cnt = 0
    missing_cnt = 0
    for filename in filenames:

        print '%i\t / \t%i' % (cnt, numOf_file)

        patient = import_json(filename)
        discharge_durgs = patient['discharge medications']

        if not discharge_durgs:
            print 'empty discharge_drug field in %s, total %i' % (filename, missing_cnt)
            missing_cnt += 1
            continue

        Y = get_med(discharge_durgs, numerical_indexed=is_number_index(discharge_durgs))
        X = extract_fields(patient)

        pickle.dump({'X': X, 'Y': Y}, open(joinpath(output_path, str(cnt)), 'w'))

        cnt += 1


def raw2dict(rawtext_path, output_path):
    """
    process original text into dict
    """

    filenames = glob.glob(joinpath(rawtext_path, '*.txt'))
    buffer_size = 10000
    heading_buffer = []
    numOf_files = len(filenames)

    cnt = 0
    filecnt = 0
    for filename in filenames:

        print '%i\t / \t%i' % (cnt, numOf_files)
        filecnt += 1

        # read raw text
        with open(filename) as f:
            rawtext = f.read()

        # remove [**<anything>**] placeholder in text
        rawtext = re.sub(r'\[\*\*.*\*\*\]', '', rawtext)

        # find out every line with pattern <heading>:<anything>
        m = re.findall(r'(^[a-zA-Z ]*):.*$', rawtext, flags=re.MULTILINE | re.UNICODE)

        if not m or len(m) == 0:
            print 'no match in file %s' % filename
            continue

        id = re.search(r'(\d+)\.txt', filename).group(1)
        heading_buffer.append({'ID': id, 'headings': m})

        if len(heading_buffer) >= buffer_size:
            pickle.dump(heading_buffer, open(joinpath(output_path, str(cnt)), 'w'))
            heading_buffer = []
            cnt += 1

            # print filename
            # pprint.pprint(m)
            # print '\n', '=' * 100, '\n'

    if len(heading_buffer) != 0:
        pickle.dump(heading_buffer, open(joinpath(output_path, str(cnt)), 'w'))


def foo(rawtext_analysis_path):
    filenames = glob.glob(joinpath(rawtext_analysis_path, '*'))
    heading_dict = {}

    cnt = 0
    for filename in filenames:

        heading_list = pickle.load(open(filename))

        for headings in heading_list:

            print '%i\t / \t%i' % (cnt, 59958)
            cnt += 1

            for h in headings['headings']:

                h = h.lower()

                if h in heading_dict:
                    heading_dict[h] += 1
                else:
                    heading_dict[h] = 1

    pprint.pprint(heading_dict)
    pickle.dump(heading_dict, open('./heading_dict', 'w'))


def bar(rawtext_path, containersave_path):
    filenames = allfilenames(rawtext_path)
    cnt = 0
    bad_cnt = 0

    for i, filename in enumerate(filenames):

        # read raw text
        with open(filename) as f:

            print '%i\t / \t%i - \t%i' % (cnt, 59958, bad_cnt), filename
            cnt += 1

            rawtext = f.read()
            try:
                NoteContainer(rawtext).save(joinpath(containersave_path, str(i)))
            except ValueError:
                traceback.print_exc()
                bad_cnt += 1
                with open('log', 'a') as logf:
                    print >> logf, filename


def bar2(container_path):
    filenames = allfilenames(container_path)
    cnt = 0

    id_cnter = {}
    cate_cnter = {}
    des_cnter = {}

    for i, filename in enumerate(filenames):

        print '%i\t / \t%i' % (cnt, 59958), filename
        cnt += 1

        nc = NoteContainer(filename, mode=1)

        id_ls = nc.get_subID()
        cate_ls, des_ls = zip(*nc.get_notetype())

        for e in id_ls:
            if e in id_cnter:
                id_cnter[e][0] += 1
            else:
                id_cnter[e] = [1, 0]

        for e in set(id_ls):
            id_cnter[e][1] += 1

        for e in cate_ls:
            if e in cate_cnter:
                cate_cnter[e][0] += 1
            else:
                cate_cnter[e] = [1, 0]

        for e in set(cate_ls):
            cate_cnter[e][1] += 1


        for e in des_ls:
            if e in des_cnter:
                des_cnter[e][0] += 1
            else:
                des_cnter[e] = [1, 0]

        for e in set(des_ls):
            des_cnter[e][1] += 1

    pprint.pprint(sorted([(k, v) for k, v in id_cnter.iteritems()], key=lambda x:x[1][1], reverse=True)[:50])
    pprint.pprint(sorted([(k, v) for k, v in cate_cnter.iteritems()], key=lambda x: x[1][1], reverse=True)[:10])
    pprint.pprint(sorted([(k, v) for k, v in des_cnter.iteritems()], key=lambda x: x[1][1], reverse=True)[:10])

    print len(id_cnter)

    pickle.dump(id_cnter, open('id_cnter', 'w'))
    pickle.dump(cate_cnter, open('cate_cnter', 'w'))
    pickle.dump(des_cnter, open('des_cnter', 'w'))


def bar3(container_path):
    filenames = allfilenames(container_path)
    cnt = 0

    id_cnter = {}

    for i, filename in enumerate(filenames):

        print '%i\t / \t%i' % (cnt, 59958), filename
        cnt += 1

        nc = NoteContainer(filename, mode=1)

        id_ls = nc.get_admID()
        cate_ls, des_ls = zip(*nc.get_notetype())

        ds_cnter = Counter(cate_ls)

        if '"Discharge summary"' in ds_cnter:

            for e in id_ls:
                if e in id_cnter:
                    id_cnter[e].append(filename)
                else:
                    id_cnter[e] = [filename]

    pprint.pprint(sorted([(k, v) for k, v in id_cnter.iteritems()], key=lambda x: len(x[1]), reverse=True)[:50])

    print len(id_cnter)

    pickle.dump(id_cnter, open('id_cnter', 'w'))


def bar4():
    id_cnter = pickle.load(open('id_cnter'))

    nc_ls = [NoteContainer(filename, mode=1) for filename in id_cnter['13033']]

    nc_ls = sorted(nc_ls, key=lambda x:map(int, x.ID_fields['CHARTDATE'].split['-']))

    pprint.pprint([nc.ID_fields for nc in nc_ls])

    print '='*80

    for nc in nc_ls:
        print nc.note[:500]


def bar5(output_path):

    id_cnter = pickle.load(open('id_cnter'))
    cnt = 0
    dup_nc = 0

    for admID, filenames in id_cnter.iteritems():

        print '%i\t / \t%i \t%i \t' % (cnt, 52690, dup_nc), admID
        cnt += 1

        nc_ls = [NoteContainer(filename, mode=1) for filename in filenames]
        nc = nc_ls[0]

        if len(nc_ls) > 1:
            dup_nc += 1
            nc.extnotes = [extnc.note for extnc in nc_ls[1:]]

        nc.save(joinpath(output_path, admID))


def bar6(uni_con_path):
    filenames = allfilenames(uni_con_path)
    heading_dict = Counter()

    for i, filename in enumerate(filenames):
        print '%i\t / \t%i' % (i, 52690), filename

        nc = NoteContainer(filename, mode=1)

        try:
            nc.extract_headings()
            heading_dict.update(nc.raw_headings)
        except RuntimeError:
            traceback.print_exc()
            with open('heading_log','a') as f:
                print >> f, filename

    pickle.dump(heading_dict, open('heading_dict', 'w'))

    pprint.pprint(sorted([(k,v) for k,v in heading_dict.iteritems()], key=lambda x:x[1], reverse=True)[:50])
    print 'total length:', len(heading_dict)


def bar7(uni_con_path, uni_con_tmp):
    global synonym
    filenames = allfilenames(uni_con_path)
    content_dict = Counter([k for k in synonym])

    for i, filename in enumerate(filenames):
        print '%i\t / \t%i' % (i, 52682), filename

        nc = NoteContainer(filename, mode=1)
        nc.extract_contents()
        content_dict.update([k for k in synonym if nc.__dict__[k] is None])

        nc.save(joinpath(uni_con_tmp, nc.get_admID()))

    pprint.pprint(content_dict)


def bar8(uni_con_tmp):
    filenames = allfilenames(uni_con_tmp)
    import numpy as np
    inds = np.random.randint(0, len(filenames), 10)

    for ind in inds:
        nc = NoteContainer(filenames[ind], mode=1)

        print '*'*80
        print '*'*36, nc.get_admID(), '*'*36
        print '*' * 80, '\n'

        print nc.note

        print '='*50

        for k in synonym:

            print k,':'

            if not nc.__dict__[k]:
                print 'NULL'
                print '-' * 40
                continue

            for v in nc.__dict__[k]:
                print v

            print '-'*40

        print '\n\n\n'


def bar9(uni_con_tmp):

    filenames = allfilenames(uni_con_tmp)

    complaint_dict = Counter()

    for i, filename in enumerate(filenames):
        print '%i\t / \t%i' % (i, 52682), filename

        nc = NoteContainer(filename, mode=1)
        if nc.chief_complaint:
            complaint_dict[nc.chief_complaint[0].strip().lower()] += 1

    pprint.pprint(complaint_dict.most_common(50))
    pickle.dump(complaint_dict, open('complaint_dict', 'w'))


def overlap_byage(uni_con_tmp):

    filenames = allfilenames(uni_con_tmp)

    med_reg = re.compile(r'^([a-z]+)[^0-9]*([0-9]+)$', flags=re.IGNORECASE)
    med_set = set()
    with open('all_meds_freq') as f:
        for line in f:
            m = med_reg.search(line)
            if not m:
                raise RuntimeError(line)

            if int(m.group(2)) < 1000:
                break

            med_set.add(m.group(1))

    # med_bigset = set()
    # with open('all_meds_freq') as f:
    #     for line in f:
    #         m = med_reg.search(line)
    #         if not m:
    #             raise RuntimeError(line)
    #
    #         if int(m.group(2)) < 50:
    #             break
    #
    #         med_bigset.add(m.group(1))

    age_cnt = Counter()
    res = []

    for i, filename in enumerate(filenames):
        print '%i\t / \t%i' % (i, 52682), filename

        nc = NoteContainer(filename, mode=1)

        try:
            nc.extract_age()
        except RuntimeError:
            continue


        age = int(nc.age)
        age_cnt[age] += 1

        add_str = ''.join(nc.admission_medications).lower() if nc.admission_medications else ''
        dis_str = ''.join(nc.discharge_medications).lower() if nc.discharge_medications else ''

        add_set = set([med for med in med_set if med in add_str])
        dis_set = set([med for med in med_set if med in dis_str])

        # # TODO debug
        # add_bigset = set([med for med in med_bigset if med in add_str])
        # dis_bigset = set([med for med in med_bigset if med in dis_str])
        #
        # print age,':'
        # print '-'*36, 'small', '-'*36
        # print add_set
        # print dis_set
        # print '-'*36, 'large', '-'*36
        # print add_bigset
        # print dis_bigset

        if len(add_set) + len(dis_set) == 0:
            res.append([age, 0.0])
            continue

        jaccard_overlap = len(add_set.intersection(dis_set)) / float(len(add_set.union(dis_set)))
        res.append([age, jaccard_overlap])

        # if len(add_bigset) + len(dis_bigset) == 0:
        #     big_lap = 0.0
        # else:
        #     big_lap = len(add_bigset.intersection(dis_bigset)) / float(len(add_bigset.union(dis_bigset)))

        # print '%.4f \t / \t %.4f' % (jaccard_overlap, big_lap)
        # print '=' * 80, '\n'

    print sum([v for v in age_cnt.itervalues()])
    pprint.pprint(age_cnt.most_common(10))

    pickle.dump(age_cnt, open('age_cnt', 'w'))
    pickle.dump(res, open('res', 'w'))


def overlap_bycomplaint(uni_con_tmp):

    filenames = allfilenames(uni_con_tmp)

    med_reg = re.compile(r'^([a-z]+)[^0-9]*([0-9]+)$', flags=re.IGNORECASE)
    med_set = set()
    with open('all_meds_freq') as f:
        for line in f:
            m = med_reg.search(line)
            if not m:
                raise RuntimeError(line)

            if int(m.group(2)) < 1000:
                break

            med_set.add(m.group(1))

    res = []

    for i, filename in enumerate(filenames):
        print '%i\t / \t%i' % (i, 52682), filename

        nc = NoteContainer(filename, mode=1)

        try:
            nc.extract_age()
        except RuntimeError:
            continue

        add_str = ''.join(nc.admission_medications).lower() if nc.admission_medications else ''
        dis_str = ''.join(nc.discharge_medications).lower() if nc.discharge_medications else ''

        add_set = set([med for med in med_set if med in add_str])
        dis_set = set([med for med in med_set if med in dis_str])

        if not nc.chief_complaint:
            continue

        complaint = nc.chief_complaint[0].strip().lower()

        if len(add_set) + len(dis_set) == 0:
            res.append([complaint, 0.0])
            continue

        jaccard_overlap = len(add_set.intersection(dis_set)) / float(len(add_set.union(dis_set)))
        res.append([complaint, jaccard_overlap])

    pickle.dump(res, open('res', 'w'))


def foo2():
    import matplotlib.pyplot as plt
    import numpy as np

    res = pickle.load(open('../data/examples/container_no_content/res'))
    age_cnt = pickle.load(open('../data/examples/container_no_content/age_cnt'))

    fig, ax1 = plt.subplots()


    age_ls = sorted([(k,v) for k,v in age_cnt.iteritems() if k < 100], key=lambda x:x[0])
    x,y = zip(*age_ls)

    ax1.plot(x, y, 'b-.')
    ax1.set_xlabel('age')

    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('population', color='b')
    ax1.tick_params('y', colors='b')


    res_dict = {}
    for age, score in res:
        if age in res_dict:
            res_dict[age].append(score)
        else:
            res_dict[age] = [score]

    res_seq = sorted([(k, np.mean(v)) for k,v in res_dict.iteritems() if k < 100], key=lambda x:x[0])
    x,y = zip(*res_seq)

    ax2 = ax1.twinx()
    ax2.plot(x, y, 'r-')
    ax2.set_ylabel('overlap score', color='r')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    plt.show()


def foo2_com():
    import matplotlib.pyplot as plt
    import numpy as np

    res = pickle.load(open('../data/examples/container_no_content/res'))
    com_dict = pickle.load(open('../data/examples/container_no_content/complaint_dict'))

    fig, ax1 = plt.subplots()


    com_ls = sorted([(k,v) for k,v in com_dict.iteritems() if v > 60], key=lambda x:x[1], reverse=True)
    x_ticks,y = zip(*com_ls)
    x = range(len(x_ticks))

    ax1.plot(x, y, 'b-.')
    ax1.set_xlabel('type')
    plt.xticks(x, x_ticks, rotation=40, ha='right')
    plt.setp(ax1.get_xticklabels(), fontsize=12)

    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('frequency', color='b')
    ax1.tick_params('y', colors='b')


    res_dict = {}
    for com, score in res:
        if com in res_dict:
            res_dict[com].append(score)
        else:
            res_dict[com] = [score]

    res_seq = sorted([(k, np.mean(v)) for k,v in res_dict.iteritems() if com_dict[k] > 60], key=lambda x:x[1], reverse=True)
    _,y = zip(*res_seq)

    ax2 = ax1.twinx()
    ax2.plot(x, y, 'r-')
    ax2.set_ylabel('overlap score', color='r')
    ax2.tick_params('y', colors='r')
    # ax2.xticks(x_ticks, rotation=40, ha='right')

    fig.tight_layout()
    plt.show()


def gen_label_index(container_dir, med_dict_path, numOf_label=50):
    filenames = allfilenames(container_dir)
    if os.path.isfile('label_index'):
        os.remove('label_index')

    med_reg = re.compile(r'^([a-z]+)[^0-9]*([0-9]+)$', flags=re.IGNORECASE)
    med_ls = []
    cnt = 0

    with open(med_dict_path) as f:

        for line in f:
            m = med_reg.search(line)

            if not m:
                raise RuntimeError(line)

            med_ls.append(m.group(1))

            cnt += 1
            if cnt >= numOf_label:
                break

    for i, filename in enumerate(filenames):
        print '%i\t / \t%i' % (i, 52682), filename

        nc = NoteContainer(filename, mode=1)

        med_str = ''.join(nc.discharge_medications).lower() if nc.discharge_medications else ''
        med_mask = [str(1 if med in med_str else 0) for med in med_ls]

        with open('label_index', 'a') as f:
            print >> f, nc.get_admID(), ' '.join(med_mask)



def makedir(_path, remove_old=False):
    if os.path.isdir(_path):
        if not remove_old:
            raise Exception('old folder exists at %s please use remove_old flag to remove' % _path)
        shutil.rmtree(_path)

    os.mkdir(_path)


if __name__ == '__main__':
    # json_filepath = '../data/notes_json_all_fields'
    # output_path = '../data/data_as_dict_tmp'
    #
    # main(json_filepath, output_path)



    # rawtext_path = 'notes/'

    output_path = 'uni_containers_tmp/'
    # output_path = '../data/examples/data_as_con/'

    # container_path = '../data/examples/container_no_content/'
    container_path = 'uni_containers/'


    # bar7(container_path, output_path)

    # bar8(output_path)
    # bar9(output_path)

    # overlap_byage(output_path)
    # overlap_bycomplaint(output_path)

    gen_label_index(output_path, 'all_meds_freq')

    # foo2()
    # foo2_com()

    # bar6(output_path)


    # if os.path.isfile('log'):
    #     os.remove('log')
    #
    # makedir(output_path, remove_old=True)
    # bar(rawtext_path, output_path)



    # makedir(output_path, True)
    # raw2dict(rawtext_path, output_path)

    # rawtext_analysis_path = '../data/examples/rawtext_analysis'
    # foo(rawtext_analysis_path)

    # heading_dict = pickle.load(open('../data/heading_dict'))
    # heading_dict_stripped = {}
    #
    # heading_list = [(k.strip(), v) for k, v in heading_dict.iteritems()]
    #
    # for k, v in heading_list:
    #     if k in heading_dict_stripped:
    #         heading_dict_stripped[k] += v
    #     else:
    #         heading_dict_stripped[k] = v
    #
    # pickle.dump(heading_dict_stripped, open('../data/heading_dict_stripped', 'w'))

    # bar3(container_path)
    # bar5(output_path)

    # bar4()

    # for filename in allfilenames(output_path):
    #     nc = NoteContainer(filename, mode=1)
    #     print nc.ID_fields





