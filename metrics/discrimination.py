import warnings 
import numpy as np

from .utils import estimate_ipcw

epsilon = 1e-4

def auc_td(e_test, t_test, risk_predicted_test, times, t, km = None, competing_risk = 1):
    """
        Calculate the time dependent AUC for competing risks.
        From the paper "Estimating and comparing time-dependent 
        areas under receiveroperating characteristic curves 
        for censored event times with competing risks"

        DOES NOT ACCOUNT FOR TIES
    """
    index = np.argmin(np.abs(times - t)) # Find horizon closest to time to eval
    km = estimate_ipcw(km)

    event = ((e_test == competing_risk) & (t_test <= t))

    if event.sum() == 0:
        return np.nan, km
   
    # Compute impact of event after t = number of event with smaller proba weighted
    after = t_test > t # Consider all event after
    if after.sum() > 0:
        after_sort = np.argsort(risk_predicted_test[after][:, index]) # Sort by risk to only have to find index to have total number

        correct_after = np.searchsorted(risk_predicted_test[:, index][after][after_sort], risk_predicted_test[event][:, index]) 
        weights_after = np.clip(km.survival_function_at_times(t_test[event]).values * km.survival_function_at_times(t).values[0], epsilon, None) if km is not None else np.array(1)

        nominator_after = (correct_after / weights_after).sum() # Total number of events with their weights
        denominator_after = after.sum() * (1 / weights_after).sum() # Total potential at risk
    else:
        nominator_after, denominator_after = 0, 0
        
    # Compute impact of competing risk before = number of event with smaller proba weighted
    before = (t_test <= t) & (e_test != competing_risk) & (e_test != 0) # Account for competing risk prior to t
    if before.sum() > 0:
        before_sort = np.argsort(risk_predicted_test[before][:, index]) # Sort by risk to only have to find index to have total number

        weights = 1. / np.clip(km.survival_function_at_times(t_test[before][before_sort]).values, epsilon, None) if km is not None else np.ones(len(before_sort))
        weighted_correct_after = np.cumsum(weights)
        total_before = np.searchsorted(risk_predicted_test[:, index][before][before_sort], risk_predicted_test[event][:, index])

        nominator_before = weighted_correct_after[total_before - 1].sum() # -1 because the sum at the index before contains the sum
        denominator_before = weighted_correct_after[-1] * before.sum() # Max weights * total potential at risk
    else:
        nominator_before, denominator_before = 0, 0

    return (nominator_after + nominator_before) / (denominator_after + denominator_before), km

def cumulative_dynamic_auc(e_test, t_test, risk_predicted_test, times, t_eval = None, km = None, competing_risk = 1):
    """
        Numerical integral of the previous auc_td
    """
    km = estimate_ipcw(km)
    t_eval = times if t_eval is None else t_eval
    aucs = [auc_td(e_test, t_test, risk_predicted_test, times, t, km, competing_risk)[0] for t in t_eval]
    t_eval, aucs = t_eval[~np.isnan(aucs)], np.array(aucs)[~np.isnan(aucs)]

    if t_eval.shape[0] < 2:
        raise ValueError("At least two time points must be given")

    return np.trapz(aucs, t_eval) / (t_eval[-1] - t_eval[0]), km

def truncated_concordance_td(e_test, t_test, risk_predicted_test, times, t, km = None, competing_risk = 1, tied_tol = 1e-8):
    """
        Compute the truncated concordance_td (no reweighting)

        ACCOUNTING FOR TIES
    """
    index = np.argmin(np.abs(times - t))
    km = estimate_ipcw(km)

    event = ((e_test == competing_risk) & (t_test <= t))
    tot_event = event.sum()

    if tot_event == 0:
        return np.nan, km
    
    if km is not None:
        weights_event = np.clip(km.survival_function_at_times(t_test), epsilon, None)
    
    nominator, discriminator = 0, 0
    for t_i, risk_predicted_i, w_i in zip(t_test[event], risk_predicted_test[event][:, index], weights_event[event]):
        after = t_test > t_i # Consider all event after
        before = (t_test <= t_i) & (e_test != competing_risk) & (e_test != 0) # Account for competing risk prior to t
        at_risk = risk_predicted_test[:, index] < risk_predicted_i
        at_risk = at_risk.astype(float)
        at_risk[np.abs(risk_predicted_test[:, index] - risk_predicted_i) <= tied_tol] = 0.5

        if km is not None:
            after = after.astype(float) / (w_i ** 2)
            before = before.astype(float) / (w_i * weights_event)
        
        nominator += ((after + before) * at_risk).sum()
        discriminator += (after + before).sum()
    return nominator / discriminator, km