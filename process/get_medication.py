import glob
import json
import io
from collections import Counter


def import_json(filename):
    with open(filename) as data_file:
        data = json.load(data_file)
    return data


def is_number_index(note):
    lines = note.split('\r')
    has_digit = 0
    for line in lines:
        if line[:3] == str(has_digit + 1) + '. ':
            has_digit += 1
        if has_digit >= 3:
            return True
    return False


indicators = [' capsule sig', ' liquid sig', ' tablet, chewable sig', ' capsule', ' gel sig', ' packet sig',
              ' misc sig',
              ' pads, medicated sig', ' tablet, chewable', ' ointment sig', ' powder sig', ' cream sig', ' sig:',
              ' powder :', ' syringe sig', ' kit sig', 'po daily', 'mg tablet', 'tablet by mouth', 'by mouth',
              ' applied t.i.d. p.r.n.', ' to the groin t.i.d. and p.r.n.', '+ d ',
              'tablet sig', ' one', ' two', 'twice', 'once', 'three', ' four', ' five', ' six', ' seven', ' eight',
              ' nine', ' ten', ' eleven', ' twelve', ' thirteen', ' fourteen', ' fifteen', ' sixteen', ' seventeen',
              ' eighteen', ' nineteen', ' twenty', ' thirty', ' fourty', 'a day', 'provider', 'tablet, delayed release',
              'daily', 'delayed release', 'release', ' day',
              'p.r.n', 'p.o.', 'p.d', 't.i.d.', 'q.i.d.', 'p.o.', 'b.i.d.', ' prn', 'q.d.', ' nf', ' q.', ' nph',
              'p.o. q.', ' ml', ' po q', ' mdi', ' mg', ' md', 'q.a.m', 'q.o.d.', ' m.d', ' patch q day', ' pr'
              ]
noise = ['nph', 'day', 'a day', 'at home', 'home o', 'a day)', 'bedtime)', 'to', 'by mouth daily', 'times a day',
         'one', 'provider)   dosage uncertain', 'by', 'pain', 'day)', 'release', 'times a day)', 'every', 'other',
         'delayed release', 'dosage uncertain', 'mouth', 'home meds', 'meds', 'daily', 'mg', 'for', 'dr', 'prn',
         'disease', ]


def rm_punc(line):
    # remove punctuations at the end of a line
    line = line.strip()
    while line and sum(line[-1] == punc for punc in ['.', '#', '%', ',', ':']) > 0:
        line = line[:-1].strip()
    return line


def get_med_name(line):
    # select any parts that are before numbers or indicator words
    i = 0
    while i < len(line):
        if line[i].isdigit():
            return rm_punc(line[:i].strip())
        for indicator in indicators:
            if line[i:i + len(indicator)] == indicator:
                return rm_punc(line[:i].strip())
        i += 1
    return rm_punc(line.strip())


def get_med(note, fo=None, numerical_indexed=False):
    if numerical_indexed:
        meds = get_med_numericalIndexed(note, fo)
    else:
        meds = get_med_lineSeperated(note, fo)
    # remove obvious noise words
    meds = [med for med in meds if med and med not in noise]
    return meds


def get_med_lineSeperated(note, fo=None):
    lines = note.split('\r')
    meds = []
    for line in lines:
        med = get_med_name(line)
        if med: meds.append(med)
    return meds


def get_med_numericalIndexed(note, fo=None):
    lines = note.split('\r')
    meds = []
    digit = 1
    for line in lines:
        if line[:3] == str(digit) + '. ':
            digit += 1
            med = get_med_name(line[3:])
            if med: meds.append(med)
            if fo:
                fo.write(meds[-1])
                fo.write('\t , \t')
                fo.write(line + '\n')
    assert len(meds) <= digit - 1
    return meds


if __name__ == "__main__":

    # ================================
    # Get discharge medication notes
    # ================================
    disch_med_notes = []
    admis_med_notes = []
    filenames = []
    files = glob.glob('../notes_json_all_fields/*.json')
    # fo = io.open('../medication_output/check_medication', 'w')
    assert len(files) > 0
    count = 0
    for file in files:
        print(file)
        patient = import_json(file)
        if not patient:
            print('[INFO] file has no content : ', file)
        if patient['discharge medications']:
            # fo.write(patient['discharge medications'])
            disch_med_notes.append(patient['discharge medications'])
            filenames.append(file)
            count += 1
            # fo.write(u'\n\n-----------------------------------------------------\n\n')
    # fo.close()
    print('Total number of patients having medication field: ', count)

    # ================================
    # Get admission medication notes
    # ================================
    admis_med_notes = []
    filenames = []
    files = glob.glob('../notes_json_all_fields/*.json')
    fo = io.open('../medication_output/admis_med_notes', 'w')
    assert len(files) > 0
    count = 0
    for file in files:
        print(file)
        patient = import_json(file)
        if not patient: print('[INFO] file has no content : ', file)
        if patient['medications on admission']:
            fo.write(u'\n\n------------- {} ----------------\n'.format(file))
            fo.write(patient['medications on admission'])
            admis_med_notes.append(patient['medications on admission'])
            filenames.append(file)
            count += 1
    # fo.close()
    print('Total number of patients having medication field: ', count)

    # ===============================================
    # Extract discharge medication names from notes
    # ===============================================
    ncount = 0
    disch_meds = {}
    fo = open('../medication_output/extracted_disch_meds', 'w')
    for filename, note in zip(filenames, disch_med_notes):
        note_meds = get_med(note, numerical_indexed=is_number_index(note))
        for med in note_meds:
            if med not in disch_meds:
                disch_meds[med] = 1
            else:
                disch_meds[med] += 1

        fo.write('\n------------ {} -----------\n'.format(filename))
        fo.write(note + '\n')
        fo.write('\n[extracted medications are] : ' + ';\t'.join(note_meds) + '\n')
    fo.close()
    print('disch_meds: ', len(disch_meds))

    # ==========================================
    # Extract admission medications from notes
    # ==========================================
    # notes = open('../check_medication', 'r').read().split('\n\n-----------------------------------------------------\n\n')
    ncount = 0
    admis_meds = {}
    fo = open('../medication_output/extracted_admis_meds', 'w')
    for filename, note in zip(filenames, admis_med_notes):
        if is_number_index(note):
            note_meds = get_med_numericalIndexed(note)
        else:
            note_meds = get_med_lineSeperated(note)

        for med in note_meds:
            if med not in admis_meds:
                admis_meds[med] = 1
            else:
                admis_meds[med] += 1

        fo.write('\n------------ {} -----------\n'.format(filename))
        fo.write(note + '\n')
        fo.write('\n[extracted medications are] : ' + '; '.join(note_meds) + '\n')

    fo.close()
    print(len(admis_med_notes), ncount)
    print('admis_meds: ', len(admis_meds))

    # ===============================
    # Record medication frequencies
    # ===============================
    from collections import defaultdict

    all_meds = defaultdict(int)
    for med in disch_meds:
        all_meds[med] += disch_meds[med]
    for med in admis_meds:
        all_meds[med] += admis_meds[med]

    fo = open('../medication_output/all_meds_freq', 'w')
    count_freq5 = 0
    count_totalfreq5 = 0
    for med, freq in reversed(sorted(all_meds.iteritems(), key=lambda (k, v): (v, k))):
        if len(med) == 1: continue
        fo.write(med + ', \t' + str(freq) + '\n')
        if freq >= 5:
            count_freq5 += 1
            count_totalfreq5 += freq
    fo.close()
    print('Total meds : ', str(len(all_meds)), '\t freq >= 5 : ', str(count_freq5))
    print('All meds : ', str(len(all_meds)), '\t all freq >= 5 : ', str(count_totalfreq5), '\t persentage : ',
          str(float(count_totalfreq5) / float(len(all_meds))))
