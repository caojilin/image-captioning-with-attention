from nlgeval import compute_metrics

metrics_dict = compute_metrics(hypothesis='eval_files/hyp.txt',
                               references=['eval_files/ref1.txt', 'eval_files/ref2.txt',
                                           'eval_files/ref3.txt', 'eval_files/ref4.txt',
                                           'eval_files/ref5.txt'])