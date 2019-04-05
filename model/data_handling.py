from numba import jit
import numpy as np


@jit(nopython=True)
def port(cons_, cat_, firm_, data_held, data_value, tick, PM):
    """
    cons_, cat_, firm_: whichi consumers are porting to which categories in which firms
    data_held: (tick, consumer, category, firm, datatype) : 1/0
    PM: (cons_, cat, firm, datatype)
    """
    # Note that data value is not decayed for the firm receiving it
    _, n_categories, n_firms, n_datatypes = data_value.shape
    # loop over every port to be made
    for i in np.arange(len(cons_)):
        cons, cat_to, firm_to = cons_[i], cat_[i], firm_[i]
        # loop over all elements in PM that are relevant
        for i_firm in np.arange(n_firms):
            for i_cat in np.arange(n_categories):
                for i_dt in np.arange(n_datatypes):
                    if PM[i, i_cat, i_firm, i_dt] == 1:
                        # there might be data to be ported!
                        counter = 0
                        for i_t in np.arange(tick):
                            if data_held[i_t, cons, cat_to, firm_to, i_dt] == 0:
                                # only port data/add to data_value if it's not already held
                                data_held[i_t, cons, cat_to, firm_to, i_dt] = data_held[
                                    i_t, cons, i_cat, i_firm, i_dt
                                ]
                                counter += data_held[i_t, cons, i_cat, i_firm, i_dt]
                        data_value[cons, cat_to, firm_to, i_dt] += counter
    return data_held, data_value


@jit(nopython=True)
def numba_calc_avail_now(requestable, A_where_0, A_where_1):
    """
    caluclates: requestable_now = requestable * A[:, :, None, None, None] * A[None, None, :, :, None]
    """
    s0, s1, s2, s3, s4 = requestable.shape
    k = len(A_where_0)
    out = np.zeros_like(requestable, dtype=np.int8)
    for i in np.arange(k):
        for j in np.arange(k):
            for i_dt in np.arange(s4):
                if (
                    requestable[
                        A_where_0[i], A_where_1[i], A_where_0[j], A_where_1[j], i_dt
                    ]
                    == 1
                ):
                    out[
                        A_where_0[i], A_where_1[i], A_where_0[j], A_where_1[j], i_dt
                    ] = 1
    return out


@jit(nopython=True)
def numba_mask_impossible_requests(r_ct, c_ct, c_f, c_cf, c_dt, requestable_now):
    """
    creates a mask of requests that are legitamate according to requestable_now.
    """
    out = np.zeros_like(r_ct, dtype=np.int8)
    for i in np.arange(len(r_ct)):
        if requestable_now[r_ct[i], c_ct[i], c_f[i], c_cf[i], c_dt[i]] == 1:
            out[i] = 1
    return out


def calculate_granting_probs(r_ct, c_f, A, low, hi):
    """
    calculates the probability of data request being granted, based on the number of overlapping
    categories the requesting and granting firm have
    """
    frac_overlap = np.nan_to_num((A[r_ct] * A[c_f]).sum(axis=-1) / A[c_f].sum(axis=1))
    return low + (hi - low) * frac_overlap


@jit(nopython=True)
def numba_update_requestable(requestable, r_ct, c_ct, c_f, c_cf, c_dt):
    """
    calculates: requestable *= (1 - request)
    """
    for a, b, c, d, e in zip(r_ct, c_ct, c_f, c_cf, c_dt):
        requestable[a, b, c, d, e] = 0
    return requestable


@jit(nopython=True)
def numba_update_portability_matrix(
    portability_matrix, r_ct_g, c_ct_g, c_f_g, c_cf_g, c_dt_g
):
    """
    updating the probability matrix
    """
    for a, b, c, d, e in zip(r_ct_g, c_ct_g, c_f_g, c_cf_g, c_dt_g):
        portability_matrix[a, b, c, d, e] = 1
    return portability_matrix


@jit(nopython=True)
def update_data_stuff(
    data_held,
    data_value,
    usage_cons,
    usage_cat,
    usage_firm,
    category_datatype,
    tick,
    n_dt,
):
    """
    update data_held and data_value after consumers having purchased  a product in the current tick
    """
    for i in np.arange(len(usage_cat)):
        cons, cat, firm = usage_cons[i], usage_cat[i], usage_firm[i]
        for j_dt in np.arange(n_dt):
            if category_datatype[cat, j_dt] == 1:
                data_held[tick, cons, cat, firm, j_dt] = 1
                data_value[cons, cat, firm, j_dt] += 1
    return data_held, data_value
