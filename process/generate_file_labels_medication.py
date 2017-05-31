import glob
import json
import numpy as np
from get_medication import get_med, is_number_index


def import_json(filename):
    with open(filename) as data_file:
        data = json.load(data_file)
    return data


# =========================================
# Get dictionary of discharge medications
# =========================================

# output: {name:index, name:index}
medications = {}
f_med = open('../medication_output/all_meds_freq', 'r').read().split('\n')
max_num_samples = 0

for i, line in enumerate(f_med):

    if not line:
        continue

    parts = line.split(', \t')
    freq = int(parts[1].strip())
    if i < 30:
        medications[parts[0].strip()] = len(medications)
        max_num_samples += freq

print('=====>>> Number of selected medications is {}, maximum number of sampels is {}'.format(len(medications),
                                                                                              max_num_samples))
num_classes = len(medications)


# =========================================
# Get dictionary of admission medications
# =========================================

# output: {name:index, name:index}
admis_meds = {}
f_admisMed = open('../medication_output/all_meds_freq', 'r').read().split('\n')
max_num_admisMedNotes = 0

for i, line in enumerate(f_admisMed):

    if not line:
        continue

    parts = line.split(', \t')
    freq = int(parts[1].strip())

    if freq >= 100:
        admis_meds[parts[0].strip()] = len(medications)
        max_num_admisMedNotes += freq

print(
    '=====>>> Number of selected medications is {}, maximum number of samples with admission medications is {}'.format(
        len(admis_meds), max_num_admisMedNotes))

num_admis_classes = len(admis_meds)


# ====================================================
# Generate file_label file for discharge medications
# ====================================================

multiclass = 1
dismed_labels = {}

files = glob.glob('../notes_json_all_fields/*.json')
assert len(files) > 0

foname = '../medication_output/file_label_0-30' + ('' if multiclass else '_1class')
fo = open(foname, 'w')

print('=====>>> Generating {}'.format(foname))



file_count = 0
sparse_count = {}

for file in files:
    patient = import_json(file)

    if patient['discharge medications']:
        note_meds = get_med(patient['discharge medications'],
                            numerical_indexed=is_number_index(patient['discharge medications']))
        note_vec = sorted([medications[med] for med in note_meds if med in medications])

        if not note_vec:
            continue

        dismed_labels[file.split('/')[-1]] = note_vec

        vec_one_hot = np.zeros(num_classes)
        for v in note_vec:
            vec_one_hot[v] = 1

        if multiclass or np.sum(vec_one_hot) == 1:
            fo.write(file.replace('../', './') + ', ' + ' '.join([str(n) for n in vec_one_hot]) + '\n')
            file_count += 1

        assert vec_one_hot.shape[0] == num_classes and len([str(n) for n in vec_one_hot]) == num_classes

        if np.sum(vec_one_hot) not in sparse_count:
            sparse_count[np.sum(vec_one_hot)] = 1
        else:
            sparse_count[np.sum(vec_one_hot)] += 1

        fo.close()
        print('=====>>> Total number of sampels for discharge medication is {}'.format(file_count))
        print('=====>>> Sparsity analysis: {}'.format(sparse_count))



# ====================================================
# Generate file_label file for admission medications
# ====================================================

admed_labels = {}
files = glob.glob('../notes_json_all_fields/*.json')

foname = '../medication_output/admisMeds_file_label_freqLe100'
fo = open(foname, 'w')

print('=====>>> Generating {}'.format(foname))

assert len(files) > 0


file_count = 0
sparse_count = {}

for file in files:
    patient = import_json(file)

    if patient['medications on admission']:
        note_meds = get_med(patient['medications on admission'],
                            numerical_indexed=is_number_index(patient['medications on admission']))
        note_vec = sorted([medications[med] for med in note_meds if med in medications])

        if not note_vec:
            continue

        admed_labels[file.split('/')[-1]] = note_vec


# vec_one_hot = np.zeros(num_admis_classes)
#         for v in note_vec:
#             vec_one_hot[v] = 1
#         fo.write(file.replace('../', './') + ', ' + ' '.join([str(n) for n in vec_one_hot]) + '\n')
#         file_count += 1
#         assert vec_one_hot.shape[0] == num_admis_classes and len([str(n) for n in vec_one_hot]) == num_admis_classes
#         if np.sum(vec_one_hot) not in sparse_count:
#             sparse_count[np.sum(vec_one_hot)] = 1
#         else:
#             sparse_count[np.sum(vec_one_hot)] += 1

# fo.close()
# print('=====>>> Total number of sampels for admission medication is {}'.format(file_count))
# print('=====>>> Sparsity analysis: {}'.format(sparse_count))



# ==============================================
# compute medication jaccard similarity scores
# ==============================================
def jaccard(veci, vecj):
    intersection = len(set(veci) & set(vecj))
    union = len(veci) + len(vecj) - intersection
    return float(intersection) / float(union)


print('=====>>> start to compute jaccard index')
num_dismed = len(dismed_labels)
print('=====>>> number of admissions is ', num_dismed)

import os

# if os.path.exists('../medication_output/jaccard_index'):
#     fo = open('../medication_output/jaccard_index', 'a')
#     fo.write('----------- start to write ----------- ')
# else:
#     fo = open('../medication_output/jaccard_index', 'w')
#
# for i, admi_i in enumerate(admi_labels):
#     if i % 1000 == 0:
#         print('[INFO] save to file at : ', i)
#         fo.close()
#         fo = open('../medication_output/jaccard_index', 'a')
#     for admi_j in admi_labels:
#         veci = admi_labels[admi_i]
#         vecj = admi_labels[admi_j]
#         score = jaccard(veci, vecj)
#         fo.write(str(admi_i) + ',\t' + str(admi_j) + ',\t' + str(score) + '\n')
#
# fo.close()


jaccard_scores = {}
jaccard_scores1 = {}  # if both dischard meds and admission meds are non-empty

for filename, dismed_label in dismed_labels.items():
    if filename in admed_labels:
        jaccard_scores1[filename] = jaccard_scores[filename] = jaccard(dismed_label, admed_labels[filename])
    else:
        jaccard_scores[filename] = jaccard(dismed_label, [])

hist = np.histogram(jaccard_scores.values())
hist1 = np.histogram(jaccard_scores1.values())

print('=====>>> histogram of jaccard scores is {}'.format(hist))
print('=====>>> histogram of jaccard1 scores is {}'.format(hist1))
print('=====>>> mean of jaccard scores is {}, number of samples is {}'.format(np.mean(jaccard_scores.values()),
                                                                              len(jaccard_scores)))
print('=====>>> mean of jaccard scores1 is {}, number of samples is {}'.format(np.mean(jaccard_scores1.values()),
                                                                               len(jaccard_scores1)))
