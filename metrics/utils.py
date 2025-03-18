from lifelines import KaplanMeierFitter

def estimate_ipcw(km):
    if isinstance(km, tuple):
        kmf = KaplanMeierFitter()
        e_train, t_train = km
        kmf.fit(t_train, e_train == 0)
        if (e_train == 0).sum() == 0:
            kmf = None
    else: kmf = km
    return kmf