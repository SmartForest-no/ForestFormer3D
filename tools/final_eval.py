from pathlib import Path
import glob
from collections import defaultdict
from plyfile import PlyData, PlyElement
import numpy as np
from scipy import stats
import os

#This file produces stats about the total average F1 score, the average F1 score per forest region, and packs all F1 score within a forest region together
#and save these stats in a file called "Eval_F1_per_region"
if __name__ == '__main__':
    import sys
    test_sem_path = sys.argv[1]
    #initialization
    NUM_CLASSES = 3  # @Treeins: classes unclassified, non-tree and tree
    NUM_CLASSES_sem = 4
    NUM_CLASSES_count = 3  # @Treeins: 2 classes without unclassified
    # class index for instance segmenatation
    ins_classcount = [2]  # @Treeins
    # class index for stuff segmentation
    stuff_classcount = [1]  # @Treeins
    # class index for semantic segmenatation
    sem_classcount = [1, 2, 3] # @Treeins
    sem_classcount_have = []
    stuff_classes = [1]
    thing_classes = [2,3]
    # Initialize...
    #test_sem_path = 'work_dirs/oneformer3d_outputfolder_continue'
    
    LOG_FOUT = open(test_sem_path + '/evaluation_total_test.txt', 'a')  # @Treeins: save evaluation file with name output_file_name

    def log_string(out_str, file=None):
        if file:
            file.write(out_str + '\n')
            file.flush()
        else:
            LOG_FOUT.write(out_str + '\n')
            LOG_FOUT.flush()
        print(out_str)

    true_positive_classes_global = np.zeros(NUM_CLASSES_sem)
    positive_classes_global = np.zeros(NUM_CLASSES_sem)
    gt_classes_global = np.zeros(NUM_CLASSES_sem)

    total_gt_ins_global = np.zeros(NUM_CLASSES)
    tpsins_global = [[] for _ in range(NUM_CLASSES)]
    fpsins_global = [[] for _ in range(NUM_CLASSES)]
    IoU_Tp_global = np.zeros(NUM_CLASSES)
    IoU_Mc_global = np.zeros(NUM_CLASSES)

    all_mean_cov_global = [[] for _ in range(NUM_CLASSES)]
    all_mean_weighted_cov_global = [[] for _ in range(NUM_CLASSES)]

    ply_files = sorted(glob.glob(test_sem_path + '/*.ply', recursive=False))

    for ply_file in ply_files:
        true_positive_classes = np.zeros(NUM_CLASSES_sem)
        positive_classes = np.zeros(NUM_CLASSES_sem)
        gt_classes = np.zeros(NUM_CLASSES_sem)

        total_gt_ins = np.zeros(NUM_CLASSES)
        tpsins = [[] for _ in range(NUM_CLASSES)]
        fpsins = [[] for _ in range(NUM_CLASSES)]
        IoU_Tp = np.zeros(NUM_CLASSES)
        IoU_Mc = np.zeros(NUM_CLASSES)

        all_mean_cov = [[] for _ in range(NUM_CLASSES)]
        all_mean_weighted_cov = [[] for _ in range(NUM_CLASSES)]

        data = PlyData(text=True).read(ply_file)

        ##################wythan wood#################################
        #ins_gt_i_ori = data.elements[0].data["instance_gt"]
        #semantic_gt = np.ones_like(ins_gt_i_ori) 

        #semantic_gt[ins_gt_i_ori == 0] = 0

        # 更新 `data.elements[0].data["semantic_gt"]`
        #data.elements[0].data["semantic_gt"] = semantic_gt
        ##################wythan wood#################################

        sem_pre_i = data.elements[0].data["semantic_pred"] + 1
        sem_gt_i = data.elements[0].data["semantic_gt"] + 1
        #sem_pre_i = data.elements[0].data["semantic_prediction_label"] + 1
        #sem_gt_i = data.elements[0].data["semantic_labels"] + 1

        ins_pre_i_ori = data.elements[0].data["instance_pred"]
        ins_gt_i_ori = data.elements[0].data["instance_gt"]
        #ins_pre_i_ori = data.elements[0].data["instance_preds"]
        #ins_gt_i_ori = data.elements[0].data["instance_labels"]


        pred_sem_complete = sem_pre_i
        gt_sem_complete = sem_gt_i
        pred_ins_complete = ins_pre_i_ori
        gt_ins_complete = ins_gt_i_ori

        idxc = ((gt_sem_complete != 0) & (gt_sem_complete != 1)) | ((pred_sem_complete != 0) & (pred_sem_complete != 1))
        pred_ins = pred_ins_complete[idxc]
        gt_ins = gt_ins_complete[idxc]
        pred_sem = pred_sem_complete[idxc]
        gt_sem = gt_sem_complete[idxc]

        for j in range(gt_sem_complete.shape[0]):
            gt_l = int(gt_sem_complete[j])
            pred_l = int(pred_sem_complete[j])
            gt_classes[gt_l] += 1
            positive_classes[pred_l] += 1
            true_positive_classes[gt_l] += int(gt_l == pred_l)

        predicted_labels_copy = pred_sem_complete.copy()
        for i in stuff_classes:
            pred_sem_complete[predicted_labels_copy == i] = 1
        for i in thing_classes:
            pred_sem_complete[predicted_labels_copy == i] = 2

        gt_labels_copy = gt_sem_complete.copy()
        for i in stuff_classes:
            gt_sem_complete[gt_labels_copy == i] = 1
        for i in thing_classes:
            gt_sem_complete[gt_labels_copy == i] = 2

        true_positive_classes_bi = np.zeros(NUM_CLASSES)
        positive_classes_bi = np.zeros(NUM_CLASSES)
        gt_classes_bi = np.zeros(NUM_CLASSES)
        for j in range(gt_sem_complete.shape[0]):
            gt_l = int(gt_sem_complete[j])
            pred_l = int(pred_sem_complete[j])
            gt_classes_bi[gt_l] += 1
            positive_classes_bi[pred_l] += 1
            true_positive_classes_bi[gt_l] += int(gt_l == pred_l)

        predicted_labels_copy = pred_sem.copy()
        for i in stuff_classes:
            pred_sem[predicted_labels_copy == i] = 1
        for i in thing_classes:
            pred_sem[predicted_labels_copy == i] = 2

        gt_labels_copy = gt_sem.copy()
        for i in stuff_classes:
            gt_sem[gt_labels_copy == i] = 1
        for i in thing_classes:
            gt_sem[gt_labels_copy == i] = 2

        un = np.unique(pred_ins)
        pts_in_pred = [[] for _ in range(NUM_CLASSES)]
        for g in un:
            if g == -1:
                continue
            tmp = (pred_ins == g)
            sem_seg_i = int(stats.mode(pred_sem[tmp])[0])
            pts_in_pred[sem_seg_i] += [tmp]

        un = np.unique(gt_ins)
        pts_in_gt = [[] for _ in range(NUM_CLASSES)]
        for g in un:
            if g == -1:
                continue
            tmp = (gt_ins == g)
            sem_seg_i = int(stats.mode(gt_sem[tmp])[0])
            pts_in_gt[sem_seg_i] += [tmp]

        for i_sem in range(NUM_CLASSES):
            sum_cov = 0
            mean_cov = 0
            mean_weighted_cov = 0
            num_gt_point = 0
            if not pts_in_gt[i_sem] or not pts_in_pred[i_sem]:
                all_mean_cov[i_sem].append(0)
                all_mean_weighted_cov[i_sem].append(0)
                continue
            for ins_gt in pts_in_gt[i_sem]:
                ovmax = 0.
                num_ins_gt_point = np.sum(ins_gt)
                num_gt_point += num_ins_gt_point
                for ins_pred in pts_in_pred[i_sem]:
                    union = (ins_pred | ins_gt)
                    intersect = (ins_pred & ins_gt)
                    iou = float(np.sum(intersect)) / np.sum(union)

                    if iou > ovmax:
                        ovmax = iou

                sum_cov += ovmax
                mean_weighted_cov += ovmax * num_ins_gt_point

            if len(pts_in_gt[i_sem]) != 0:
                mean_cov = sum_cov / len(pts_in_gt[i_sem])
                all_mean_cov[i_sem].append(mean_cov)

                mean_weighted_cov /= num_gt_point
                all_mean_weighted_cov[i_sem].append(mean_weighted_cov)

        for i_sem in range(NUM_CLASSES):
            if not pts_in_pred[i_sem]:
                continue
            IoU_Tp_per = 0
            IoU_Mc_per = 0
            tp = [0.] * len(pts_in_pred[i_sem])
            fp = [0.] * len(pts_in_pred[i_sem])
            if pts_in_gt[i_sem]:
                total_gt_ins[i_sem] += len(pts_in_gt[i_sem])
            for ip, ins_pred in enumerate(pts_in_pred[i_sem]):
                ovmax = -1.
                if not pts_in_gt[i_sem]:
                    fp[ip] = 1
                    continue
                for ins_gt in pts_in_gt[i_sem]:
                    union = (ins_pred | ins_gt)
                    intersect = (ins_pred & ins_gt)
                    iou = float(np.sum(intersect)) / np.sum(union)

                    if iou > ovmax:
                        ovmax = iou

                if ovmax > 0:
                    IoU_Mc_per += ovmax
                if ovmax >= 0.5:
                    tp[ip] = 1  # true
                    IoU_Tp_per += ovmax
                else:
                    fp[ip] = 1  # false positive

            tpsins[i_sem] += tp
            fpsins[i_sem] += fp
            IoU_Tp[i_sem] += IoU_Tp_per
            IoU_Mc[i_sem] += IoU_Mc_per

        file_name = os.path.basename(ply_file).split('.')[0]
        individual_log_path = os.path.join(test_sem_path, f'{file_name}_evaluation_test.txt')
        with open(individual_log_path, 'w') as IND_LOG_FOUT:
            # semantic results
            iou_list = []
            sem_classcount_have = []
            for i in range(NUM_CLASSES_sem):
                if gt_classes[i] > 0:
                    sem_classcount_have.append(i)
                    iou = true_positive_classes[i] / float(gt_classes[i] + positive_classes[i] - true_positive_classes[i])
                else:
                    iou = 0.0
                iou_list.append(iou)

            set1 = set(sem_classcount)
            set2 = set(sem_classcount_have)
            set3 = set1 & set2
            sem_classcount_final = list(set3)

            log_string('Semantic Segmentation oAcc: {}'.format(sum(true_positive_classes) / float(sum(positive_classes))), IND_LOG_FOUT)
            log_string('Semantic Segmentation mAcc: {}'.format(np.mean(true_positive_classes[sem_classcount_final] / gt_classes[sem_classcount_final])), IND_LOG_FOUT)
            log_string('Semantic Segmentation IoU: {}'.format(iou_list), IND_LOG_FOUT)
            log_string('Semantic Segmentation mIoU: {}'.format(1. * sum(iou_list) / len(sem_classcount_final)), IND_LOG_FOUT)
            log_string('  ', IND_LOG_FOUT)

            iou_list_bi = []
            sem_classcount_have_bi = []
            for i in range(NUM_CLASSES):
                if gt_classes_bi[i] > 0:
                    sem_classcount_have_bi.append(i)
                    iou = true_positive_classes_bi[i] / float(gt_classes_bi[i] + positive_classes_bi[i] - true_positive_classes_bi[i])
                else:
                    iou = 0.0
                iou_list_bi.append(iou)

            sem_classcount_bi = [1, 2]
            set1 = set(sem_classcount_bi)
            set2 = set(sem_classcount_have_bi)
            set3 = set1 & set2
            sem_classcount_final_bi = list(set3)

            set1 = set(stuff_classcount)
            set2 = set(sem_classcount_have_bi)
            set3 = set1 & set2
            stuff_classcount_final = list(set3)

            log_string('Binary Semantic Segmentation oAcc: {}'.format(sum(true_positive_classes_bi) / float(sum(positive_classes_bi))), IND_LOG_FOUT)
            log_string('Binary Semantic Segmentation mAcc: {}'.format(np.mean(true_positive_classes_bi[sem_classcount_final_bi] / gt_classes_bi[sem_classcount_final_bi])), IND_LOG_FOUT)
            log_string('Binary Semantic Segmentation IoU: {}'.format(iou_list_bi), IND_LOG_FOUT)
            log_string('Binary Semantic Segmentation mIoU: {}'.format(1. * sum(iou_list_bi) / len(sem_classcount_final_bi)), IND_LOG_FOUT)
            log_string('  ', IND_LOG_FOUT)

            MUCov = np.zeros(NUM_CLASSES)
            MWCov = np.zeros(NUM_CLASSES)
            for i_sem in range(NUM_CLASSES):
                MUCov[i_sem] = np.mean(all_mean_cov[i_sem])
                MWCov[i_sem] = np.mean(all_mean_weighted_cov[i_sem])

            precision = np.zeros(NUM_CLASSES)
            recall = np.zeros(NUM_CLASSES)
            RQ = np.zeros(NUM_CLASSES)
            SQ = np.zeros(NUM_CLASSES)
            PQ = np.zeros(NUM_CLASSES)
            PQStar = np.zeros(NUM_CLASSES)
            set1 = set(ins_classcount)
            set2 = set(sem_classcount_have)
            set3 = set1 & set2
            ins_classcount_final = list(set3)

            for i_sem in ins_classcount:
                if not tpsins[i_sem] or not fpsins[i_sem]:
                    continue
                tp = np.asarray(tpsins[i_sem]).astype(float)
                fp = np.asarray(fpsins[i_sem]).astype(float)
                tp = np.sum(tp)
                fp = np.sum(fp)
                if total_gt_ins[i_sem] == 0:
                    rec = 0
                else:
                    rec = tp / total_gt_ins[i_sem]
                if (tp + fp) == 0:
                    prec = 0
                else:
                    prec = tp / (tp + fp)
                precision[i_sem] = prec
                recall[i_sem] = rec
                if (prec + rec) == 0:
                    RQ[i_sem] = 0
                else:
                    RQ[i_sem] = 2 * prec * rec / (prec + rec)
                if tp == 0:
                    SQ[i_sem] = 0
                else:
                    SQ[i_sem] = IoU_Tp[i_sem] / tp
                PQ[i_sem] = SQ[i_sem] * RQ[i_sem]
                PQStar[i_sem] = PQ[i_sem]

            for i_sem in stuff_classcount:
                if iou_list_bi[i_sem] >= 0.5:
                    RQ[i_sem] = 1
                    SQ[i_sem] = iou_list_bi[i_sem]
                else:
                    RQ[i_sem] = 0
                    SQ[i_sem] = 0
                PQ[i_sem] = SQ[i_sem] * RQ[i_sem]
                PQStar[i_sem] = iou_list_bi[i_sem]

            if np.mean(precision[ins_classcount_final]) + np.mean(recall[ins_classcount_final]) == 0:
                F1_score = 0.0
            else:
                F1_score = (2 * np.mean(precision[ins_classcount_final]) * np.mean(recall[ins_classcount_final])) / (
                            np.mean(precision[ins_classcount_final]) + np.mean(recall[ins_classcount_final]))

            log_string('Instance Segmentation for Offset:', IND_LOG_FOUT)
            log_string('Instance Segmentation MUCov: {}'.format(MUCov[ins_classcount]), IND_LOG_FOUT)
            log_string('Instance Segmentation mMUCov: {}'.format(np.mean(MUCov[ins_classcount_final])), IND_LOG_FOUT)
            log_string('Instance Segmentation MWCov: {}'.format(MWCov[ins_classcount]), IND_LOG_FOUT)
            log_string('Instance Segmentation mMWCov: {}'.format(np.mean(MWCov[ins_classcount_final])), IND_LOG_FOUT)
            log_string('Instance Segmentation Precision: {}'.format(precision[ins_classcount]), IND_LOG_FOUT)
            log_string('Instance Segmentation mPrecision: {}'.format(np.mean(precision[ins_classcount_final])), IND_LOG_FOUT)
            log_string('Instance Segmentation Recall: {}'.format(recall[ins_classcount]), IND_LOG_FOUT)
            log_string('Instance Segmentation mRecall: {}'.format(np.mean(recall[ins_classcount_final])), IND_LOG_FOUT)
            log_string('Instance Segmentation F1 score: {}'.format(F1_score), IND_LOG_FOUT)
            log_string('Instance Segmentation RQ: {}'.format(RQ[sem_classcount_bi]), IND_LOG_FOUT)
            log_string('Instance Segmentation meanRQ: {}'.format(np.mean(RQ[sem_classcount_final_bi])), IND_LOG_FOUT)
            log_string('Instance Segmentation SQ: {}'.format(SQ[sem_classcount_bi]), IND_LOG_FOUT)
            log_string('Instance Segmentation meanSQ: {}'.format(np.mean(SQ[sem_classcount_final_bi])), IND_LOG_FOUT)
            log_string('Instance Segmentation PQ: {}'.format(PQ[sem_classcount_bi]), IND_LOG_FOUT)
            log_string('Instance Segmentation meanPQ: {}'.format(np.mean(PQ[sem_classcount_final_bi])), IND_LOG_FOUT)
            log_string('Instance Segmentation PQ star: {}'.format(PQStar[sem_classcount_bi]), IND_LOG_FOUT)
            log_string('Instance Segmentation mean PQ star: {}'.format(np.mean(PQStar[sem_classcount_final_bi])), IND_LOG_FOUT)
            log_string('Instance Segmentation RQ (things): {}'.format(RQ[ins_classcount]), IND_LOG_FOUT)
            log_string('Instance Segmentation meanRQ (things): {}'.format(np.mean(RQ[ins_classcount_final])), IND_LOG_FOUT)
            log_string('Instance Segmentation SQ (things): {}'.format(SQ[ins_classcount]), IND_LOG_FOUT)
            log_string('Instance Segmentation meanSQ (things): {}'.format(np.mean(SQ[ins_classcount_final])), IND_LOG_FOUT)
            log_string('Instance Segmentation PQ (things): {}'.format(PQ[ins_classcount]), IND_LOG_FOUT)
            log_string('Instance Segmentation meanPQ (things): {}'.format(np.mean(PQ[ins_classcount_final])), IND_LOG_FOUT)
            log_string('Instance Segmentation RQ (stuff): {}'.format(RQ[stuff_classcount]), IND_LOG_FOUT)
            log_string('Instance Segmentation meanRQ (stuff): {}'.format(np.mean(RQ[stuff_classcount_final])), IND_LOG_FOUT)
            log_string('Instance Segmentation SQ (stuff): {}'.format(SQ[stuff_classcount]), IND_LOG_FOUT)
            log_string('Instance Segmentation meanSQ (stuff): {}'.format(np.mean(SQ[stuff_classcount_final])), IND_LOG_FOUT)
            log_string('Instance Segmentation PQ (stuff): {}'.format(PQ[stuff_classcount]), IND_LOG_FOUT)
            log_string('Instance Segmentation meanPQ (stuff): {}'.format(np.mean(PQ[stuff_classcount_final])), IND_LOG_FOUT)

        true_positive_classes_global += true_positive_classes
        positive_classes_global += positive_classes
        gt_classes_global += gt_classes

        total_gt_ins_global += total_gt_ins
        for i in range(NUM_CLASSES):
            tpsins_global[i] += tpsins[i]
            fpsins_global[i] += fpsins[i]
            IoU_Tp_global[i] += IoU_Tp[i]
            IoU_Mc_global[i] += IoU_Mc[i]

        for i in range(NUM_CLASSES):
            all_mean_cov_global[i] += all_mean_cov[i]
            all_mean_weighted_cov_global[i] += all_mean_weighted_cov[i]

    iou_list_global = []
    sem_classcount_have_global = []
    for i in range(NUM_CLASSES_sem):
        if gt_classes_global[i] > 0:
            sem_classcount_have_global.append(i)
            iou_global = true_positive_classes_global[i] / float(gt_classes_global[i] + positive_classes_global[i] - true_positive_classes_global[i])
        else:
            iou_global = 0.0
        iou_list_global.append(iou_global)

    set1_global = set(sem_classcount)
    set2_global = set(sem_classcount_have_global)
    set3_global = set1_global & set2_global
    sem_classcount_final_global = list(set3_global)

    log_string('Semantic Segmentation oAcc: {}'.format(sum(true_positive_classes_global) / float(sum(positive_classes_global))))
    log_string('Semantic Segmentation mAcc: {}'.format(np.mean(true_positive_classes_global[sem_classcount_final_global] / gt_classes_global[sem_classcount_final_global])))
    log_string('Semantic Segmentation IoU: {}'.format(iou_list_global))
    log_string('Semantic Segmentation mIoU: {}'.format(1. * sum(iou_list_global) / len(sem_classcount_final_global)))
    log_string('  ')

    iou_list_bi_global = []
    sem_classcount_have_bi_global = []
    for i in range(NUM_CLASSES):
        if gt_classes_bi[i] > 0:
            sem_classcount_have_bi_global.append(i)
            iou_bi_global = true_positive_classes_bi[i] / float(gt_classes_bi[i] + positive_classes_bi[i] - true_positive_classes_bi[i])
        else:
            iou_bi_global = 0.0
        iou_list_bi_global.append(iou_bi_global)

    sem_classcount_bi_global = [1, 2]
    set1_bi_global = set(sem_classcount_bi_global)
    set2_bi_global = set(sem_classcount_have_bi_global)
    set3_bi_global = set1_bi_global & set2_bi_global
    sem_classcount_final_bi_global = list(set3_bi_global)

    set1_stuff_global = set(stuff_classcount)
    set2_stuff_global = set(sem_classcount_have_bi_global)
    set3_stuff_global = set1_stuff_global & set2_stuff_global
    stuff_classcount_final_global = list(set3_stuff_global)

    log_string('Binary Semantic Segmentation oAcc: {}'.format(sum(true_positive_classes_bi) / float(sum(positive_classes_bi))))
    log_string('Binary Semantic Segmentation mAcc: {}'.format(np.mean(true_positive_classes_bi[sem_classcount_final_bi_global] / gt_classes_bi[sem_classcount_final_bi_global])))
    log_string('Binary Semantic Segmentation IoU: {}'.format(iou_list_bi_global))
    log_string('Binary Semantic Segmentation mIoU: {}'.format(1. * sum(iou_list_bi_global) / len(sem_classcount_final_bi_global)))
    log_string('  ')

    MUCov_global = np.zeros(NUM_CLASSES)
    MWCov_global = np.zeros(NUM_CLASSES)
    for i_sem in range(NUM_CLASSES):
        MUCov_global[i_sem] = np.mean(all_mean_cov_global[i_sem])
        MWCov_global[i_sem] = np.mean(all_mean_weighted_cov_global[i_sem])

    precision_global = np.zeros(NUM_CLASSES)
    recall_global = np.zeros(NUM_CLASSES)
    RQ_global = np.zeros(NUM_CLASSES)
    SQ_global = np.zeros(NUM_CLASSES)
    PQ_global = np.zeros(NUM_CLASSES)
    PQStar_global = np.zeros(NUM_CLASSES)
    set1_ins_global = set(ins_classcount)
    set2_ins_global = set(sem_classcount_have_global)
    set3_ins_global = set1_ins_global & set2_ins_global
    ins_classcount_final_global = list(set3_ins_global)

    for i_sem in ins_classcount:
        if not tpsins_global[i_sem] or not fpsins_global[i_sem]:
            continue
        tp_global = np.asarray(tpsins_global[i_sem]).astype(float)
        fp_global = np.asarray(fpsins_global[i_sem]).astype(float)
        tp_global = np.sum(tp_global)
        fp_global = np.sum(fp_global)
        if total_gt_ins_global[i_sem] == 0:
            rec_global = 0
        else:
            rec_global = tp_global / total_gt_ins_global[i_sem]
        if (tp_global + fp_global) == 0:
            prec_global = 0
        else:
            prec_global = tp_global / (tp_global + fp_global)
        precision_global[i_sem] = prec_global
        recall_global[i_sem] = rec_global
        if (prec_global + rec_global) == 0:
            RQ_global[i_sem] = 0
        else:
            RQ_global[i_sem] = 2 * prec_global * rec_global / (prec_global + rec_global)
        if tp_global == 0:
            SQ_global[i_sem] = 0
        else:
            SQ_global[i_sem] = IoU_Tp_global[i_sem] / tp_global
        PQ_global[i_sem] = SQ_global[i_sem] * RQ_global[i_sem]
        PQStar_global[i_sem] = PQ_global[i_sem]

    for i_sem in stuff_classcount:
        if iou_list_bi_global[i_sem] >= 0.5:
            RQ_global[i_sem] = 1
            SQ_global[i_sem] = iou_list_bi_global[i_sem]
        else:
            RQ_global[i_sem] = 0
            SQ_global[i_sem] = 0
        PQ_global[i_sem] = SQ_global[i_sem] * RQ_global[i_sem]
        PQStar_global[i_sem] = iou_list_bi_global[i_sem]

    if np.mean(precision_global[ins_classcount_final_global]) + np.mean(recall_global[ins_classcount_final_global]) == 0:
        F1_score_global = 0.0
    else:
        F1_score_global = (2 * np.mean(precision_global[ins_classcount_final_global]) * np.mean(recall_global[ins_classcount_final_global])) / (
                    np.mean(precision_global[ins_classcount_final_global]) + np.mean(recall_global[ins_classcount_final_global]))

    log_string('Instance Segmentation for Offset:')
    log_string('Instance Segmentation MUCov: {}'.format(MUCov_global[ins_classcount]))
    log_string('Instance Segmentation mMUCov: {}'.format(np.mean(MUCov_global[ins_classcount_final_global])))
    log_string('Instance Segmentation MWCov: {}'.format(MWCov_global[ins_classcount]))
    log_string('Instance Segmentation mMWCov: {}'.format(np.mean(MWCov_global[ins_classcount_final_global])))
    log_string('Instance Segmentation Precision: {}'.format(precision_global[ins_classcount]))
    log_string('Instance Segmentation mPrecision: {}'.format(np.mean(precision_global[ins_classcount_final_global])))
    log_string('Instance Segmentation Recall: {}'.format(recall_global[ins_classcount]))
    log_string('Instance Segmentation mRecall: {}'.format(np.mean(recall_global[ins_classcount_final_global])))
    log_string('Instance Segmentation F1 score: {}'.format(F1_score_global))
    log_string('Instance Segmentation RQ: {}'.format(RQ_global[sem_classcount_bi_global]))
    log_string('Instance Segmentation meanRQ: {}'.format(np.mean(RQ_global[sem_classcount_final_bi_global])))
    log_string('Instance Segmentation SQ: {}'.format(SQ_global[sem_classcount_bi_global]))
    log_string('Instance Segmentation meanSQ: {}'.format(np.mean(SQ_global[sem_classcount_final_bi_global])))
    log_string('Instance Segmentation PQ: {}'.format(PQ_global[sem_classcount_bi_global]))
    log_string('Instance Segmentation meanPQ: {}'.format(np.mean(PQ_global[sem_classcount_final_bi_global])))
    log_string('Instance Segmentation PQ star: {}'.format(PQStar_global[sem_classcount_bi_global]))
    log_string('Instance Segmentation mean PQ star: {}'.format(np.mean(PQStar_global[sem_classcount_final_bi_global])))
    log_string('Instance Segmentation RQ (things): {}'.format(RQ_global[ins_classcount]))
    log_string('Instance Segmentation meanRQ (things): {}'.format(np.mean(RQ_global[ins_classcount_final_global])))
    log_string('Instance Segmentation SQ (things): {}'.format(SQ_global[ins_classcount]))
    log_string('Instance Segmentation meanSQ (things): {}'.format(np.mean(SQ_global[ins_classcount_final_global])))
    log_string('Instance Segmentation PQ (things): {}'.format(PQ_global[ins_classcount]))
    log_string('Instance Segmentation meanPQ (things): {}'.format(np.mean(PQ_global[ins_classcount_final_global])))
    log_string('Instance Segmentation RQ (stuff): {}'.format(RQ_global[stuff_classcount]))
    log_string('Instance Segmentation meanRQ (stuff): {}'.format(np.mean(RQ_global[stuff_classcount_final_global])))
    log_string('Instance Segmentation SQ (stuff): {}'.format(SQ_global[stuff_classcount]))
    log_string('Instance Segmentation meanSQ (stuff): {}'.format(np.mean(SQ_global[stuff_classcount_final_global])))
    log_string('Instance Segmentation PQ (stuff): {}'.format(PQ_global[stuff_classcount]))
    log_string('Instance Segmentation meanPQ (stuff): {}'.format(np.mean(PQ_global[stuff_classcount_final_global])))

    LOG_FOUT.close()