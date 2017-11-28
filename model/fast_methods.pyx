import numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def mul_feat_coeff_with_items(int[:] users, int[:] items, double[:, :] feat, double coeff_global,
                                      double[:, :] coeff_users, double[:] coeff_items, double[:] mul_sum):
    cdef int N = feat.shape[0]
    cdef int M = feat.shape[1]
    cdef int i, j
    cdef double point_sum = 0

    with nogil:
        for i in range(N):
            point_sum = 0
            for j in range(M):
                point_sum += feat[i, j] * coeff_users[users[i], j]

            point_sum += coeff_global  # It's a 1 constant
            point_sum += coeff_items[items[i]]  # It's a 1 constant
            mul_sum[i] = point_sum

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def mul_feat_coeff_no_items(int[:] users, double[:, :] feat, double coeff_global, double[:, :] coeff_users,
                             double[:] mul_sum):
    cdef int N = feat.shape[0]
    cdef int M = feat.shape[1]
    cdef int i, j
    cdef double point_sum = 0

    with nogil:
        for i in range(N):
            point_sum = 0
            for j in range(M):
                point_sum += feat[i, j] * coeff_users[users[i], j]

            point_sum += coeff_global  # It's a 1 constant
            mul_sum[i] = point_sum

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def grad_for_user(int[:] users, double[:, :] d_u_mle, double[:, :] d_u_prior, double[:] user_counts, double[:, :] grad):
    cdef int N = d_u_mle.shape[0]
    cdef int M = d_u_mle.shape[1]
    cdef int K = user_counts.shape[0]
    cdef int i, j

    with nogil:
        for i in range(N):
            user_counts[users[i]] += 1

            # The sum of the derivative values
            for j in range(M):
                grad[users[i], j] += d_u_mle[i, j]

        # Normalizing and subtracting the prior
        for i in range(K):
            if user_counts[i] > 0:
                for j in range(M):
                    grad[i, j] /= user_counts[i]
                    grad[i, j] -= d_u_prior[i, j]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def grad_for_item(int[:] items, double[:] d_i_mle, double[:] d_i_prior, double[:] item_counts, double[:] grad):
    cdef int N = d_i_mle.shape[0]
    cdef int M = d_i_mle.shape[1]
    cdef int K = item_counts.shape[0]
    cdef int i

    with nogil:
        for i in range(N):
            item_counts[items[i]] += 1
            grad[items[i]] += d_i_mle[i]

        # Normalizing and subtracting the prior
        for i in range(K):
            if item_counts[i] > 0:
                grad[i] /= item_counts[i]
                grad[i] -= d_i_prior[i]


@cython.boundscheck(False)
@cython.wraparound(False)
def convert_rates_to_erank(double[:] rates, int[:] order, double[:] erank):
    cdef int i, j, k, ac_num
    cdef int n = rates.shape[0]
    cdef double ac_sum

    with nogil:
        i = 0
        while i < n:
            ac_sum = i
            ac_num = 1
            j = i + 1
            while j < n:
                if rates[order[i]] > rates[order[j]]:
                    break
                ac_sum += j
                ac_num += 1
                j += 1

            for k in range(i, j):
                erank[order[k]] = ac_sum / ac_num

            i = j
