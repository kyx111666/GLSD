"""Test-only real remote consumer functions from the passed stage-0 package.
Inputs are synthetic. smooth/convertLabel helpers are from local training_utils.
Production uses the actual remote training_utils module, never this fixture.
"""
import numpy as np
from numpy import argmax
from collections import Counter
from scipy.signal import find_peaks
from Utils.mean_average_precision.mean_average_precision import MeanAveragePrecision2d

def smooth(y, box_pts):
    y = [each_y for each_y in y]
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def convertLabel(label):
    label_dict = { 'negative' : 0, 'positive' : 1, 'surprise' : 2, 'others' : 3 }
    return label_dict[label]

def spotting(final_samples, subject_count, pred_interval, total_gt_spot, p, metric_final, k_p):
    pred_subject = []
    gt_subject = []
    metric_video = MeanAveragePrecision2d(num_classes=1)
    for videoIndex, video in enumerate(final_samples[subject_count]):
        preds = []
        gt = []
        score_plot = np.array(pred_interval[videoIndex])
        score_plot = smooth(score_plot, k_p*2)
        score_plot_agg = score_plot.copy()

        threshold = score_plot_agg.mean() + p * (max(score_plot_agg) - score_plot_agg.mean()) #Moilanen threshold technique
        peaks, _ = find_peaks(score_plot_agg, height=threshold, distance=k_p)
        for peak in peaks:
            preds.append([peak-k_p, 0, peak+k_p, 0, 0, 0, peak]) #Extend left and right side of peak by k frames

        if len(peaks)==0 or len(preds)==0:
            preds = np.empty((0, 7))
        for samples in video: 
            gt.append([samples[0], 0, samples[2], 0, 0, 0, 0, samples[1]])
            total_gt_spot += 1
        metric_video.add(np.array(preds),np.array(gt))
        metric_final.add(np.array(preds),np.array(gt)) #IoU = 0.5 according to MEGC2020 metrics
        pred_subject.append(preds)
        gt_subject.append(gt)

    return pred_subject, gt_subject, total_gt_spot, metric_video, metric_final


def recognition(result, preds, metric_video, final_emotions, subject_count, pred_list, gt_tp_list, final_samples, pred_window_list, pred_single_list,frame_skip):
    cur_pred = []
    cur_tp_gt = []
    pred_gt_recog = []
    cur_pred_window = []
    cur_pred_single = []
    pred_emotion = result #splitVideo(result, subject_count+1, final_samples, final_dataset_spotting) #Split predicted emotion by video
    pred_match_gt = sorted(metric_video.value(iou_thresholds=0.5)[0.5][0]['pred_match_gt'].items())
    for video_index, video_match in pred_match_gt: #key=video_index, value=match index for each video
        for pred_index, sample_index in enumerate(video_match): #pred_index=index of prediction array, sample_index=index of emotion array
            pred_onset = max(0, preds[video_index][pred_index][0])
            pred_peak = max(0, preds[video_index][pred_index][-1]) 
            pred_offset = max(0, preds[video_index][pred_index][2])
            pred_emotion_list = pred_emotion[video_index][max(0, pred_onset +1):max(1, pred_offset -1)]

            most_common_emotion, _ = Counter(pred_emotion_list).most_common(1)[0]

            cur_pred.append(most_common_emotion)
            pred_gt_recog.append(argmax(pred_emotion[video_index][final_samples[subject_count][video_index][0][0]])) #Predicted emotion on gt onset label
            gt_label = final_emotions[subject_count][video_index][sample_index] #Get video emotion    
            if(sample_index!=-1):
                cur_tp_gt.append(convertLabel(gt_label))
            else:
                cur_tp_gt.append(-1)
    pred_list.extend(cur_pred)
    gt_tp_list.extend(cur_tp_gt)
    pred_window_list.extend(cur_pred_window)
    pred_single_list.extend(cur_pred_single)
    print('Predicted with k_p     :', cur_pred)
    return pred_list, cur_pred, gt_tp_list, pred_window_list, pred_single_list


def sequence_evaluation(total_gt_spot, metric_final): #Get TP, FP, FN for final evaluation
    TP_spot = int(sum(metric_final.value(iou_thresholds=0.5)[0.5][0]['tp'])) 
    FP_spot = int(sum(metric_final.value(iou_thresholds=0.5)[0.5][0]['fp']))
    FN_spot = total_gt_spot - TP_spot
    print('TP:', TP_spot, 'FP:', FP_spot, 'FN:', FN_spot)
    return TP_spot, FP_spot, FN_spot

