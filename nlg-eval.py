import pickle
import json
from nlgeval import compute_metrics

with open('references.pickle', 'rb') as f1:
    references = pickle.load(f1)

with open('hypotheses.pickle', 'rb') as f2:
    hypotheses = pickle.load(f2)



# a = references[0:5]
# b = hypotheses[0:5]
a = references
b = hypotheses



# with open('ref1.txt', 'a') as the_file:
#     the_file.write('Hello\n')
# with open('ref2.txt', 'a') as the_file:
#     the_file.write('Hello\n')
# with open('ref3.txt', 'a') as the_file:
#     the_file.write('Hello\n')
# with open('ref4.txt', 'a') as the_file:
#     the_file.write('Hello\n')
# with open('ref5.txt', 'a') as the_file:
#     the_file.write('Hello\n')


# word_map_file = "WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json"

word_map_file = "WORDMAP_coco_5_cap_per_img_5_min_word_freq.json"
# seq = [9, 2166, 12, 3, 4, 59, 67, 1, 983, 231]


# Load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}

# words = [rev_word_map[ind] for ind in seq]


for count, elem in enumerate(b):
    with open('eval_files/hyp.txt', 'a') as the_file:
        words = [rev_word_map[ind] for ind in elem]
        sentence = " ".join(i for i in words)
        the_file.write('{0}\n'.format(sentence))

for i in range(len(a)):
    for count, elem in enumerate(a[i]):
        with open('eval_files/ref{0}.txt'.format(count+1), 'a') as the_file:
            words = [rev_word_map[ind] for ind in elem]
            sentence = " ".join(i for i in words)
            the_file.write('{0}\n'.format(sentence))

metrics_dict = compute_metrics(hypothesis='eval_files/hyp.txt',
                               references=['eval_files/ref1.txt', 'eval_files/ref2.txt',
                                           'eval_files/ref3.txt', 'eval_files/ref4.txt',
                                           'eval_files/ref5.txt'])
