import csv

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


class Evaluation:
    def __init__(self, save_dir, save_alldata):
        self.save_dir = save_dir
        self.save_alldata = save_alldata

    def measure_accessbiliy_prediction(self, y_label, ground_label):
        roc_auc_avg = []
        pr_auc_avg = []

        for i in range(len(y_label)):
            single_y_label = y_label[i]
            single_ground_label = ground_label[i]
            try:
                roc_auc = roc_auc_score(y_score=single_y_label,
                                        y_true=single_ground_label)
                precision, recall, _ = precision_recall_curve(probas_pred=single_y_label,
                                                              y_true=single_ground_label)
                pr_auc = auc(recall, precision)

                roc_auc_avg.append(roc_auc)
                pr_auc_avg.append(pr_auc)
            except ValueError:
                pass

        final_roc_auc = np.mean(np.array(roc_auc_avg))
        final_pr_auc = np.mean(np.array(pr_auc_avg))

        res_save = pd.DataFrame({'roc_auc': roc_auc_avg, 'pr_auc': pr_auc_avg})
        res_save.to_csv(self.save_alldata, index=False)

        with open(self.save_dir, 'w') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerows([['roc_auc', 'pr_auc'],
                            [final_roc_auc, final_pr_auc]])

    def measure_clustering(self):
        pass
