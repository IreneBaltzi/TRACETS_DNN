import numpy as np  


def _compute_auc(labels, predictions, check_errors=True):
    if check_errors:  
        if len(predictions) != len(labels):  
            raise Exception('Numbers of predictions and labels must be the same.')
        for label in labels:                 
            if label not in (0, 1):
                raise Exception('Labels must satisfy label == 0 or label == 1.')

    thresholds = np.unique(predictions)[::-1]  
    if thresholds.size == 0:                  
        thresholds = np.array([1.0, 0.0], dtype=float) 
    else:
        if thresholds[0] != 1:                 
            thresholds = np.insert(thresholds, 0, 1.0)
        if thresholds[-1] == 0:                
            thresholds = thresholds[:-1]

    n = len(labels)                            
    m = len(thresholds)                        
    tp = np.zeros(m)                           
    fp = np.zeros(m)                            
    fn = np.zeros(m)                            
    tn = np.zeros(m)                            
    idx = np.argsort(predictions)[::-1]        
    i = 0                                       

    for j in range(m):                          
        if j == 0:                              
            tp[j] = 0
            fp[j] = 0
            fn[j] = np.sum(labels)             
            tn[j] = n - fn[j]                   
        else:                                   
            tp[j] = tp[j-1]
            fp[j] = fp[j-1]
            fn[j] = fn[j-1]
            tn[j] = tn[j-1]

        while i < n and predictions[idx[i]] >= thresholds[j]:
            if labels[idx[i]]:                  
                tp[j] += 1                      
                fn[j] -= 1                      
            else:                               
                fp[j] += 1                      
                tn[j] -= 1                      
            i += 1                              

    tpr = np.zeros(m)                            
    tnr = np.zeros(m)                            
    ppv = np.zeros(m)                            

    for j in range(m):                           
        tpr[j] = tp[j] / (tp[j] + fn[j]) if (tp[j] + fn[j]) else 1.0
        tnr[j] = tn[j] / (fp[j] + tn[j]) if (fp[j] + tn[j]) else 1.0
        ppv[j] = tp[j] / (tp[j] + fp[j]) if (tp[j] + fp[j]) else 1.0

    auroc = 0.0                                  
    auprc = 0.0                                  
    for j in range(m - 1):                       
        auroc += 0.5 * (tpr[j+1] - tpr[j]) * (tnr[j+1] + tnr[j])  
        auprc += (tpr[j+1] - tpr[j]) * ppv[j+1]                   
    return float(auroc), float(auprc)            


def _compute_accuracy_f_measure(labels, predictions, check_errors=True):
    if check_errors:                              
        if len(predictions) != len(labels):       
            raise Exception('Numbers of predictions and labels must be the same.')
        for label in labels:                      
            if label not in (0, 1):
                raise Exception('Labels must satisfy label == 0 or label == 1.')

    tp = fp = fn = tn = 0                          

    for y, yhat in zip(labels, predictions):       
        if y and yhat: tp += 1                     
        elif (not y) and yhat: fp += 1             
        elif y and (not yhat): fn += 1             
        else: tn += 1                               

    denom = tp + fp + fn + tn                      
    acc = (tp + tn) / denom if denom else 1.0      

    denom_f = 2 * tp + fp + fn                     
    f1 = (2 * tp) / denom_f if denom_f else 1.0    
    return float(acc), float(f1)                   


def _compute_prediction_utility(labels, predictions,
                                dt_early=-12, dt_optimal=-6, dt_late=3,
                                max_u_tp=1, min_u_fn=-2, u_fp=-0.05, u_tn=0,
                                check_errors=True):
    if check_errors and len(predictions) != len(labels):  
        raise Exception('Numbers of predictions and labels must be the same.')

    labels = np.asarray(labels, dtype=int)        
    predictions = np.asarray(predictions, dtype=int)

    if np.any(labels):                            
        t_sepsis = int(np.argmax(labels)) - dt_optimal  
        is_septic = True                          
    else:
        t_sepsis = np.inf                         
        is_septic = False                         

    n = len(labels)                                
    m_1 = float(max_u_tp) / float(dt_optimal - dt_early); b_1 = -m_1 * dt_early
    m_2 = float(-max_u_tp) / float(dt_late - dt_optimal); b_2 = -m_2 * dt_late
    m_3 = float(min_u_fn) / float(dt_late - dt_optimal); b_3 = -m_3 * dt_optimal

    u = np.zeros(n, dtype=float)                   
    for t in range(n):                             
        if t <= t_sepsis + dt_late:                
            if is_septic and predictions[t]:       
                if t <= t_sepsis + dt_optimal:     
                    u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)  
                elif t <= t_sepsis + dt_late:      
                    u[t] = m_2 * (t - t_sepsis) + b_2             
            elif (not is_septic) and predictions[t]:  
                u[t] = u_fp                           
            elif is_septic and (not predictions[t]):  
                if t <= t_sepsis + dt_optimal:        
                    u[t] = 0
                elif t <= t_sepsis + dt_late:         
                    u[t] = m_3 * (t - t_sepsis) + b_3
            else:
                u[t] = u_tn                            
    return float(u.sum())                               


def evaluate_sepsis_from_lists(cohort_labels, cohort_preds_bin, cohort_probs):
    labels_all = np.concatenate(cohort_labels).astype(int) if len(cohort_labels) else np.array([], dtype=int)
    preds_all  = np.concatenate(cohort_preds_bin).astype(int) if len(cohort_preds_bin) else np.array([], dtype=int)
    probs_all  = np.concatenate(cohort_probs).astype(float) if len(cohort_probs) else np.array([], dtype=float)

    if len(labels_all) == 0:             
        return 0.0, 0.0, 0.0, 0.0, 0.0

    auroc, auprc = _compute_auc(labels_all, probs_all)
    accuracy, f_measure = _compute_accuracy_f_measure(labels_all, preds_all)

    n_files = len(cohort_labels)                  
    obs = np.zeros(n_files)                       
    best = np.zeros(n_files)                      
    worst = np.zeros(n_files)                     
    inact = np.zeros(n_files)                     

    dt_early, dt_optimal, dt_late = -12, -6, 3   
    max_u_tp, min_u_fn, u_fp, u_tn = 1, -2, -0.05, 0  

    for k in range(n_files):                      
        y = cohort_labels[k].astype(int)          
        yhat = cohort_preds_bin[k].astype(int)    
        num_rows = len(y)                         

        best_pred = np.zeros(num_rows, dtype=int) 
        if np.any(y):                              
            t_sepsis = int(np.argmax(y)) - dt_optimal  
            s = max(0, t_sepsis + dt_early)            
            e = min(t_sepsis + dt_late + 1, num_rows)  
            best_pred[s:e] = 1                         

        worst_pred = 1 - best_pred                 
        inaction_pred = np.zeros(num_rows, dtype=int)  

        obs[k]   = _compute_prediction_utility(y, yhat, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
        best[k]  = _compute_prediction_utility(y, best_pred, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
        worst[k] = _compute_prediction_utility(y, worst_pred, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
        inact[k] = _compute_prediction_utility(y, inaction_pred, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)

    denom = (best.sum() - inact.sum())            
    norm_utility = (obs.sum() - inact.sum()) / denom if denom != 0 else 0.0

    return float(auroc), float(auprc), float(accuracy), float(f_measure), float(norm_utility)
