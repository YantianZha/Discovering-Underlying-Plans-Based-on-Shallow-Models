import cython

def compute_correct_preds_rnn(blank_count, pediction_set_size, plan, planRNN, indices, correct_rnn):
    cdef int blank_order
    cdef int _blank_count = blank_count
    cdef int blank_index
    cdef int _correct_rnn = correct_rnn
    cdef int i

    assert len(plan) == len(planRNN[0])
    for blank_order in range(0, _blank_count):
        blank_index = indices[blank_order]
        for i in xrange(0, pediction_set_size):
            if plan[blank_index] == planRNN[i][blank_index]:
                _correct_rnn += 1
                break
    return _correct_rnn

def compute_correct_preds_dup(blank_count, indices, best_plan_args, actions, plan, correct=0):
    cdef int blank_order
    cdef int _blank_count = blank_count
    cdef int sample_index
    cdef int _correct = correct

    for blank_order in range(0, _blank_count):
        blank_index = indices[blank_order]
        for sample_index in best_plan_args[:, blank_order]:
            if actions[sample_index] == plan[blank_index]:
                _correct += 1
                break
    return _correct