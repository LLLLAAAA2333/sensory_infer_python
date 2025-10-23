# -*- coding: utf-8 -*-
# 
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com


from src.comm_utils.packages import *


def unsupervised_clustering_metrics(pred_with_label: Dict):
    labels_true = []
    labels_pred = []
    for cluster_label, true_labels in pred_with_label.items():
        for true_label in true_labels:
            labels_true.append(true_label)
            labels_pred.append(cluster_label)

    encoder = LabelEncoder()
    labels_true = encoder.fit_transform(labels_true)

    silhouette_score = .5 + .5 * metrics.silhouette_score(np.array(labels_true).reshape(-1, 1), labels_pred)
    calinski_harabasz_score = metrics.calinski_harabasz_score(np.array(labels_true).reshape(-1, 1), labels_pred)

    return silhouette_score, calinski_harabasz_score
