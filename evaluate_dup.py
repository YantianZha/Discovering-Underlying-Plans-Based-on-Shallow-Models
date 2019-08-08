#!/usr/bin/python
from gensim import models
from copy import deepcopy
from math import ceil, floor
from itertools import permutations
import random
import sys, getopt
import numpy as np
from numpy import exp, dot, log
from six.moves import cPickle
import logging
import os
import operator
import tensorflow as tf
#from word_rnn import main, sample
from word_rnn import sample
from word_rnn import model as RNNmodel
import argparse
import train_and_test_inner
import gc
import psutil

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("compute.log")
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# fm = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s [%(filename)s:%(lineno)s] - %(message)s')
# fh.setFormatter(fm)
# ch.setFormatter(fm)
logger.addHandler(fh)

lr = 0.05
iter_num = 10
proc = psutil.Process(os.getpid())
mem0 = proc.memory_info().rss
pd = lambda x2, x1: 100.0 * (x2 - x1) / mem0

def remove_random_actions(plan):
    blank_count = int(ceil(len(plan) * blank_percentage + 0.5)) # +0.5: lower bound, >0
    incomplete_plan = deepcopy(plan)
    indices = []
    cnt = 0
    while cnt < blank_count:
        missing_action_index = random.randrange(2, len(plan)-2)
        if missing_action_index in indices:
            # making sure that the indices generated are unique
            continue
        else:
            incomplete_plan[ missing_action_index ] = u'###'
            indices.append(missing_action_index)
            cnt += 1
    return blank_count, indices, incomplete_plan

def remove_randomN_actions(plan, blank_count):
    incomplete_plan = deepcopy(plan)
    indices = []
    cnt = 0
    while cnt < blank_count:
        missing_action_index = random.randrange(2, len(plan)-2)
        if missing_action_index in indices:
            # making sure that the indices generated are unique
            continue
        else:
            incomplete_plan[ missing_action_index ] = u'###'
            indices.append(missing_action_index)
            cnt += 1
    return blank_count, indices, incomplete_plan

def remove_random_actions_end(plan):
    blank_count = int(ceil(len(plan) * blank_percentage + 0.5))
    incomplete_plan = deepcopy(plan)
    indices = []
    cnt = 0
    while cnt < blank_count:
        missing_action_index = len(plan)-1-cnt
        incomplete_plan[ missing_action_index ] = u'###'
        indices.append(missing_action_index)
        cnt += 1
    return blank_count, indices, incomplete_plan0704

def remove_lastN_actions(plan, blank_count):
    incomplete_plan = deepcopy(plan)
    indices = []
    cnt = 0
    while cnt < blank_count:
        missing_action_index = len(plan)-1-cnt
        incomplete_plan[ missing_action_index ] = u'###'
        indices.append(missing_action_index)
        cnt += 1
    return blank_count, indices, incomplete_plan

def remove_conseq_middle_actions(plan, blank_count):
    incomplete_plan = deepcopy(plan)
    indices = []
    cnt = 0
    missing_action_init = random.randrange(2, len(plan) - 2 - blank_count)
    # while True:
    #     missing_action_init = random.randrange(2, len(plan)-2-blank_count)
    #     if all([i in range(len(plan)) for i in range(missing_action_init, missing_action_init + blank_count)]):
    #         break
    #     else: continue
    for i in range(missing_action_init, missing_action_init+blank_count):
        incomplete_plan[i] = u'###'
        indices.append(i)
        cnt += 1
    return blank_count, indices, incomplete_plan

# p = permutation of actions
# ip = incomplete plan
def getTentativePlan(p, ip, indices):
    for i in range(len(indices)):
        ip[indices[i]] = p[i]
    return ip

# def permuteOverMissingActions(actions, blank_count, indices):
#     ''' Exausts 64 GB of RAM when
#         blank_count >= 3,
#         #( actions ) >= 1250
#     '''
#     action_set = []
#     tentative_plans = []
#     for p in permutations(actions, blank_count):
#      action_set.append(p)
#      tentative_plans.append(getTentativePlan(p, incomplete_plan, indices))
#     return action_set, tentative_plans

# def predictAndVerify(indices, tentative_plans, action_set):
#     for i in range(len(indices)):
#         window_sized_plans = []
#         for tp in tentative_plans:
#             window_sized_plans.append( tp[indices[i]-window_size:indices[i]+window_size+1] )
#         scores = model.score( window_sized_plans )
#         best_indices = scores.argsort()[-1*pediction_set_size:][::-1]
#         for j in best_indices:
#             if action_set[j][i] == plan[indices[i]]:
#                 correct += 1
#                 break;
#     return correct

def min_uncertainty_distance_in_window_size(indices):
    # Makes sure that within a window size there is only one missing action
    # Optimized code from http://stackoverflow.com/questions/15606537/finding-minimal-difference
    if len(indices) <= window_size:
        return 2
    idx = deepcopy(indices)
    idx = sorted(idx)
    res = [ idx[i+window_size]-idx[i] for i in xrange(len(idx)) if i+window_size < len(idx) ]
    return min(res)


def score_sg_pair(model, word, word2):
    l1 = model.syn0[word2.index]
    l2a = deepcopy(model.syn1[word.point])  # 2d matrix, codelen x layer1_size
    sgn = -1.0**word.code  # ch function, 0-> 1, 1 -> -1
    lprob = -log(1.0 + exp(-sgn*dot(l1, l2a.T)))
    return sum(lprob)


def score_sg_grad_b(model, word, context_word, b, a):
    l1 = model.syn0[context_word.index] # vector of context word
    l2a = deepcopy(model.syn1[word.point])  # 2d matrix, codelen x layer1_size
    sgn = -1.0**word.code  # ch function, 0-> 1, 1 -> -1
    sigma = 1.0 / (1.0 + exp(-sgn*dot(a * l1, b * l2a.T)))   # p(context_word|word)
    grads = (1.0 - sigma) * dot(a * l1, l2a.T) * sgn  # gradient respect to parameter b
    return sum(grads)


def compute_gradient(model, blank_index, sample_word, target_weight, current_weight, incompelete_plan):
    grad = 0.0
    vocab = model.vocab
    current_word = vocab[sample_word]
    context_words = [ vocab[incompelete_plan[blank_index-1]], vocab[incompelete_plan[blank_index+1]] ]
    for target_word in context_words:
        grad += score_sg_grad_b(model, current_word, target_word, current_weight, target_weight)
        # grad += score_sg_grad(model, current_word, target_word, current_weight, target_weight)
    # print grad
    return grad


def test_grad(blank_count, model, plan, blank_index):
    tmp_plan =  deepcopy(plan)
    vocab_size = len(model.vocab.keys())
    weights = np.ones(vocab_size * blank_count).reshape(vocab_size, blank_count) / vocab_size
    gradients = np.zeros(vocab_size * blank_count).reshape(vocab_size, blank_count)
    actions = model.wv.vocab.keys()
    grad_dict = {}
    score_dict = {}
    # true_word = plan[blank_index]
    logger.debug("true_word\tsample_word\tgrad")
    for k in range(vocab_size):
        sample_index = k
        sample_word = actions[sample_index]
        current_weight = weights[sample_index][0]
        grad = compute_gradient(model, blank_index, sample_word, 1,
                                current_weight, plan)
        grad_dict[sample_word] = grad
        tmp_plan[blank_index] = sample_word
        score_dict[sample_word] = model.score([tmp_plan[blank_index-1:blank_index+2]])
        logger.debug("%s\t%s\t%s", plan[blank_index], sample_word, grad)
        gradients[sample_index][0] += grad
        # # update weights
        # weights += gradients
        # # min-max normalize to 0-1
        # mins = np.amin(weights, axis=0)
        # maxs = np.amax(weights, axis=0)
        # weights = (weights - mins) / (maxs - mins)

    sorted_x = sorted(grad_dict.items(), key=operator.itemgetter(1), reverse=True)
    order_grad = sorted_x.index([item for item in sorted_x if item[0] == plan[blank_index]][0])
    logger.info("order grad:%d", order_grad)
    logger.info("sorted grad")
    logger.info("word\tgrad\torder")
    for order, item in enumerate(sorted_x, start=1):
        if item[0] == plan[blank_index]:
            logger.info("***")
        logger.info("%s\t%f\t%d", item[0], item[1], order)

    sorted_y = sorted(score_dict.items(), key=operator.itemgetter(1), reverse=True)
    # order of true word score
    order_score = sorted_y.index([item for item in sorted_y if item[0] == plan[blank_index]][0])
    logger.info("order score:%d", order_score)
    logger.info("sorted score")
    logger.info("word\tscore\torder")
    for order, item in enumerate(sorted_y, start=1):
        if item[0] == plan[blank_index]:
            logger.info("***")
        logger.info("%s\t%f\t%d", item[0], item[1], order)


def test_pair_sg(model, target_word, current_word, target_weight, current_weight):

    score_dict = {}
    d = model.vocab
    vocab = model.vocab.keys()
    logger.info("true word:%s", current_word)
    logger.info("current_word\ttarget_word\tscore")
    for word in vocab:
        score = score_sg_grad_b(model, d[target_word], d[word], target_weight, current_weight)
        score_dict[word] = score
        if word == current_word:
            logger.info("***")
        logger.info("%s\t%s\t%f", word, target_word, score)
    sort_x = sorted(score_dict.items(), key=operator.itemgetter(1), reverse=True)
    for order, item in enumerate(sort_x, 1):
        if item[0] == current_word:
            logger.info("my score order:%d", order)
            logger.info("my score:%f", item[1])

    gensim_score_dict = {}
    # d = model.vocab
    logger.info("gensim score")
    logger.info("current_word\ttarget_word\tscore")
    for word in vocab:
        score = score_sg_pair(model, d[target_word], d[word])
        # score = score_sg_grad(model, d[target_word], d[word], target_weight, current_weight)
        gensim_score_dict[word] = score
        if word == current_word:
            logger.info("***")
        logger.info("%s\t%s\t%f", word, target_word, score)
    sort_y = sorted(gensim_score_dict.items(), key=operator.itemgetter(1), reverse=True)
    for order, item in enumerate(sort_y, 1):
        if item[0] == current_word:
            logger.info("gensim score order:%d", order)
            logger.info("gensim score:%f", item[1])

def gen_plan_rnn(model, ckpt, saver, sess, words, vocab, args):
    # dummy_args = {}
    # inc_plan_len = len(incomplete_plan)
    # dummy_args['save_dir'] = './word_rnn/save'; dummy_args['n'] = blank_count; dummy_args['prime'] = incomplete_plan[:inc_plan_len-blank_count]
    # dummy_args['sample'] = 1; dummy_args['pick'] = 1
    # if itr == 0:
    #     new_plan = sample.sampleNEW(dummy_args)
    # else:
    #     new_plan = sample.sampleNEW1(dummy_args)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        new_plan = model.sample(sess, words, vocab, args['n'], args['prime'], args['sample'], args['pick'])
    return new_plan

def train_and_test(tf_args, gen_args):
    '''
    The function trains a model on training data and then tests the models accuracy on the testing data.
    Since training is time consuming, we save the model and load it later for further testing
    '''
    domain, shouldTrain, set_number, pediction_set_size, mode, missing, biasWin, use_rnn, winSz = gen_args
    print "\n=== Set : %s ===\n" % str(set_number)

    # Train a model based on training data
    if shouldTrain == True:
        sentences = models.word2vec.LineSentence(domain+'/train/train'+str(set_number)+'.txt')
        model = models.Word2Vec(sentences=sentences, min_count=1, sg=1, workers=4, hs=1, window=winSz, iter=20, sample=0) # sg=0, cbox, 1, skipgram, default worker = 4
        model.save(domain+'/model'+str(set_number)+'.txt')
    else:
        # OR load a mode
        model = models.Word2Vec.load(domain+'/model'+str(set_number)+'.txt')

    print "Training : COMPLETE!"

    # Evaluate model on test data
    plans = open(domain+'/test/test'+str(set_number)+'.txt').read().split("\n")
    list_of_actions = [[unicode(actn, "utf-8") for actn in plan_i.split()] for plan_i in plans]
    actions = model.wv.vocab.keys()
    vocab_size = len(actions)
    correct = 0
    correct_rnn = 0
    total = 0
    total_correct_rnn_plan = 0

    print "Testing : RUNNING . . ."
    list_of_actions = [x for x in list_of_actions if len(x) != 0]

    # test compute gradient
    # test_grad(1, model, list_of_actions[0], 4)


    sess, dummy_args, words, vocab, saver, ckpt, planGen = tf_args

    for itr in xrange(len(list_of_actions)):
        planRNN = []
        plan = list_of_actions[itr]

        if mode == 'end' and 0 < missing < 1:
            blank_count, indices, incomplete_plan = remove_lastN_actions(plan, int(ceil(len(plan) * missing + 0.5)))#remove_lastN_actions(plan,1)
            indices.reverse()
        elif mode == 'end' and missing >= 1:
            blank_count, indices, incomplete_plan = remove_lastN_actions(plan, int(missing))
            indices.reverse()
        elif mode == 'middle_random' and missing >= 1:
            blank_count, indices, incomplete_plan = remove_randomN_actions(plan, int(missing))
        elif mode == 'middle_cons' and missing >= 1:
            blank_count, indices, incomplete_plan = remove_conseq_middle_actions(plan, int(missing))
        elif mode == 'middle_random' and 0 < missing < 1:
            blank_count, indices, incomplete_plan = remove_randomN_actions(plan, int(ceil(len(plan) * missing + 0.5)))
        elif mode == 'middle_cons' and 0 < missing < 1:
            blank_count, indices, incomplete_plan = remove_conseq_middle_actions(plan, int(ceil(len(plan) * missing + 0.5)))
        else:
            break
            #Todo:
        #blank_count, indices, incomplete_plan = remove_lastN_actions(plan, int(ceil(len(plan) * blank_percentage + 0.5)))#remove_lastN_actions(plan,1)
        # blank_count, indices, incomplete_plan = remove_lastN_actions(plan, 1)
        # indices.reverse()
        if use_rnn == True:

        # dummy_args['n'] = blank_count; inc_plan_len = len(incomplete_plan)
        # dummy_args['prime'] = incomplete_plan[:indices[0]]#inc_plan_len-blank_count] #middle
        # #dummy_args['prime'] = incomplete_plan[:inc_plan_len-blank_count]
        # for i in xrange(pediction_set_size):
        #     dummy_args_bp = deepcopy(dummy_args)
        #     gen = gen_plan_rnn(planGen, ckpt, saver, sess, words, vocab, dummy_args_bp)
        #     #gen = gen_plan_rnn(planGen, ckpt, saver, sess, words, vocab, dummy_args_bp) + plan[indices[0]+1:] # middle
        #     planRNN.append(gen)
        # #planRNN = gen_plan_rnn(planGen, ckpt, saver, sess, words, vocab, dummy_args)


            dummy_args['n'] = blank_count;
            inc_plan_len = len(incomplete_plan)
            dummy_args['prime'] = incomplete_plan[:indices[0]]  # inc_plan_len-blank_count] #middle
            # dummy_args['prime'] = incomplete_plan[:inc_plan_len-blank_count]
            for i in xrange(pediction_set_size):
                dummy_args_bp = deepcopy(dummy_args)
                if mode == 'end':
                    gen = gen_plan_rnn(planGen, ckpt, saver, sess, words, vocab, dummy_args_bp)
                elif (mode == 'middle_cons' or 'middle_random') and int(missing) == 1:
                    gen = gen_plan_rnn(planGen, ckpt, saver, sess, words, vocab, dummy_args_bp) + plan[indices[0] + 1:]  # middle
                elif mode == 'middle_cons' and int(missing) > 1:
                    gen = gen_plan_rnn(planGen, ckpt, saver, sess, words, vocab, dummy_args_bp) + plan[indices[-1] + 1:]  # middle
                else:
                    break
                    # Todo:
                planRNN.append(gen)

        total += blank_count
        weights = np.zeros(vocab_size * blank_count).reshape(vocab_size, blank_count)
        # weights_for_sample = np.ones(vocab_size * blank_count).reshape(vocab_size, blank_count) / vocab_size

        # random fill the blank word
        random_indices = random.sample(range(vocab_size), blank_count)
        for order in range(blank_count):
            blank_index = indices[order]
            random_word = actions[random_indices[order]]
            incomplete_plan[blank_index] = random_word

        # confirm weather incomplete_plan is updated in the last iteration
        update_flag = True
        for iter in range(iter_num):
            # save current_words to change update_flag
            current_words = []
            for index in indices:
                current_words.append(incomplete_plan[index])
            if update_flag:
                predict_words = []
                for blank_order in range(blank_count):
                    tentative_plans = []
                    blank_index = indices[blank_order]
                    for vocab_index in range(vocab_size):
                        incomplete_plan[blank_index] = actions[vocab_index]
                        # build tmp_plan for compute score
                        #tmp_plan = incomplete_plan[blank_index - window_size:blank_index + window_size + 2]
                        #tmp_plan = incomplete_plan[blank_index-window_size-1:blank_index+window_size+1]
                        #tmp_plan = incomplete_plan[blank_index - 2*winSz : blank_index + 1] # end
                        tmp_plan = incomplete_plan[blank_index - winSz + biasWin: blank_index + winSz + biasWin + 1]
                        tentative_plans.append(tmp_plan)
                    scores = model.score(tentative_plans)
                    weights[:, blank_order] = scores
                    # select word that has max score to update blank word
                    max_index = np.argmax(weights[:, blank_order])
                    predict_word = actions[max_index]
                    incomplete_plan[blank_index] = predict_word
                    predict_words.append(predict_word)
                logger.info("predict words:%s", predict_words)
            else:
                # no update in the last iteration
                logger.info("quit from no update, iteratiototal_correct_predictionsn:%d", iter)
                break
            if predict_words == current_words:
                update_flag = False
            else:
                update_flag = True
        if iter+1 == iter_num:
            logger.info("quit from reach max iteration, iteration:%d", iter)
        best_plan_args = np.argsort(weights, axis=0)[-1*pediction_set_size:][::-1]
        logger.info("best args")
        logger.info("%s", best_plan_args)
        for i in range(blank_count):
            blank_index = indices[i]
            logger.info("%d blank word:%s", i, plan[blank_index])
            logger.info("predict word\tweights")
            for j in range(pediction_set_size):
                logger.info("%s\t%f", actions[best_plan_args[j][i]], weights[best_plan_args[j][i]][i])

        # for blank_order in range(blank_count):
        #     blank_index = indices[blank_order]
        #     for sample_index in best_plan_args[:, blank_order]:
        #         #print actions[sample_index], plan[blank_index], blank_order, blank_index, itr
        #         if actions[sample_index] == plan[blank_index]:
        #             correct += 1
        #             break
        correct = train_and_test_inner.compute_correct_preds_dup(blank_count, indices, best_plan_args, actions, plan, correct)

        if use_rnn == True:
            correct_rnn = train_and_test_inner.compute_correct_preds_rnn(blank_count, pediction_set_size, plan, planRNN, indices, correct_rnn)
            # for blank_order in range(blank_count):
            #     blank_index = indices[blank_order]
            #     assert len(plan) == len(planRNN[0])
            #     for i in xrange(pediction_set_size):
            #         #print plan[blank_index], planRNN[i][blank_index]
            #         if plan[blank_index] == planRNN[i][blank_index]:
            #             correct_rnn += 1
            #             break

        # mema = proc.memory_info().rss
        del planRNN, plan
        # memb = proc.memory_info().rss
        # print "Memory clear in function: %0.2f%%" % pd(memb, mema)

        assert (correct_rnn <= total) and (correct <= total)

        # Print at certain time intervals
        if (itr*100)/len(list_of_actions) % 10 == 0:
            sys.stdout.write( "\rProgress: %s %%" % str( (itr*100)/len(list_of_actions) ) )
            sys.stdout.flush()


    # sys.stdout.write( "\r\rTesting : COMPLETE!\n")
    # sys.stdout.flush()
    # print "\nUnknown actions: %s; Correct predictions: %s" % (str(total), str(correct))
    # print "Set Accuracy: %s\n" % str( float(correct*100)/total)
    #return total, correct, correct_rnn
    return total, correct, correct_rnn, total_correct_rnn_plan

def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    return saver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_wv', type=bool, default=1, help='Need to train wordemb or not?')
    # parser.add_argument('--use_rnn', type=bool, default=False, help='If have the RNN model for testing')
    parser.add_argument('--use_rnn', action ='store_true')
    parser.add_argument('--domain', type=str, default='blocks', help='domain?')
    parser.add_argument('--mode', type=str, default='end', help='middle or not')
    # parser.add_argument('--num_missing', type=float, default=1, help='number of missing actions')
    parser.add_argument('--win_bias', type=int, default=0, help='window bias for word embedding')
    parser.add_argument('--top_k', nargs="+", type=int, help='range of size of candidate predictions: a b')
    parser.add_argument('--win_range', nargs="+", type=int, help='range of word2vec window size: a b')
    parser.add_argument('--num_missing', nargs="+", type=int, help='number of missing actions')

    args = parser.parse_args()
    #print argv
    domain = args.domain
    train = args.train_wv
    mode = args.mode
    # num_missing = args.num_missing
    biasWin = args.win_bias
    k = 1
    topkRange = tuple(args.top_k)
    winRange = tuple(args.win_range)
    num_missing_in = tuple(args.num_missing)
    use_rnn = 1#args.use_rnn # BUG!
    dir = os.path.dirname(__file__)
    rnnmodel_dir = os.path.join(dir, domain+'/rnnModel8'
                                            '')

    print "\n=== Domain : %s ===\n" % domain

    total_unknown_actions = 0
    total_correct_predictions = 0
    total_correct_rnn_predictions = 0
    total_correct_rnn_plan_recognitions = 0

    dummy_args = {}
    #inc_plan_len = len(incomplete_plan)
    dummy_args['save_dir'] = rnnmodel_dir#'./word_rnn/save';
    dummy_args['sample'] = 1; dummy_args['pick'] = 1

    with open(os.path.join(dummy_args['save_dir'], 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(dummy_args['save_dir'], 'words_vocab.pkl'), 'rb') as f:
        words, vocab = cPickle.load(f)
    planGen = RNNmodel.Model(saved_args, True)

    model_path = tf.train.latest_checkpoint(dummy_args['save_dir'])
    ckpt = tf.train.get_checkpoint_state(dummy_args['save_dir'])

    with tf.Session() as sess:
        saver = optimistic_restore(sess, model_path)
        saver.restore(sess, model_path)
        tf.global_variables_initializer().run()
        tf_args = sess, dummy_args, words, vocab, saver, ckpt, planGen
        for topk in range(topkRange[0], topkRange[1]+1):
            for winSz in range(winRange[0], winRange[1]+1):
                for num_missing in range(num_missing_in[0], num_missing_in[1]+1):

                    for i in range(k):
                        gen_args = domain, train, i, topk, mode, num_missing, biasWin, use_rnn, winSz
                        ua, cp, rnncp, tcrp = train_and_test(tf_args, gen_args)
                        total_unknown_actions += ua
                        total_correct_predictions += cp
                        total_correct_rnn_predictions += rnncp
                        total_correct_rnn_plan_recognitions += tcrp

                    print "\n==== FINAL STATISTICS ===="
                    print "topk: %d" % (topk)
                    print "window_size: %d" % (winSz)
                    print "\nTotal unknown actions: %d; Total correct predictions: %d; Total correct RNN predictions: %d" % (total_unknown_actions, total_correct_predictions, total_correct_rnn_predictions)
                    print "DUP ACCURACY: %0.2f%%\n" % (float(total_correct_predictions*100)/total_unknown_actions)
                    print "RNN ACCURACY: %0.2f%%\n" % (float(total_correct_rnn_predictions * 100) / total_unknown_actions)
                    print "topk, winSz, num_missing", topk, winSz, num_missing
                    total_unknown_actions = 0
                    total_correct_predictions = 0
                    total_correct_rnn_predictions = 0
                    total_correct_rnn_plan_recognitions = 0

                    # mem1 = proc.memory_info().rss
                    gc.collect()
                    # mem2 = proc.memory_info().rss
                    # print "Collect memory: %0.2f%%" % pd(mem2, mem1)


if __name__ == "__main__":
#    for k in xrange(2,6):
#        print "Top- %d \n", k
#        main(sys.argv[1:], k)
    main()

