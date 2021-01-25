'''
Created on March 24, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''

import torch
import torch.optim as optim

from NGCF import NGCF
from utility.helper import *
from utility.batch_test import *
from utility.process_music import *
from utility.matrix import *

import warnings
warnings.filterwarnings('ignore')
from time import time
import sys
from tqdm import tqdm

if __name__ == '__main__':
    print("Date Loading")
    args.device = torch.device('cuda:' + str(args.gpu_id))
    #plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()
    adj_matrix, u_features, v_features, test_playlists, \
    train_playlists_count, playlists_tracks = process_mpd(0, 10000)
    users_to_test = [i for i in range(10000)]
    u_features = torch.from_numpy(u_features.toarray()).to(args.device).float()
    n_users, n_items = adj_matrix.shape
    path = "../Data/mpd"
    norm_adj = get_adj_mat(n_users, n_items, adj_matrix, path)
    print(n_users, n_items, norm_adj.shape)
    exist_users = []
    train_items = []
    count_zero_pid = 0

    for i in tqdm(range(n_users)):
      train_items.append(adj_matrix[i].indices)
      if len(adj_matrix[i].indices) == 0:
        count_zero_pid += 1
        continue
      exist_users.append(i)
    
    t_n_users = n_users - count_zero_pid
    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)

    model = NGCF(n_users, n_items,
                 norm_adj,
                 args, u_features).to(args.device)

    t0 = time()
    """
    *********************************************************
    Train.
    """
    print(">>>Train")
    cur_best_pre_0, stopping_step = 0, 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.
        n_batch = n_users // args.batch_size + 1

        for idx in range(n_batch):
            users, pos_items, neg_items = sample(args.batch_size, t_n_users, \
                                                  exist_users, train_items, n_items)
            # print(len(users), len(pos_items), len(neg_items))
            # print(np.array(users).max(), np.array(pos_items).max(), np.array(neg_items).max())
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                           pos_items,
                                                                           neg_items,
                                                                           drop_flag=args.node_dropout_flag)
            #print(u_g_embeddings.shape, pos_i_g_embeddings.shape, neg_i_g_embeddings.shape)
            batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                              pos_i_g_embeddings,
                                                                              neg_i_g_embeddings)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss

        if args.verbose > 0 and epoch % args.verbose == 0:
            perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                epoch, time() - t1, loss, mf_loss, emb_loss)
            print(perf_str)

        

        #loss_loger.append(loss)
        # rec_loger.append(ret['recall'])
        # pre_loger.append(ret['precision'])
        # ndcg_loger.append(ret['ndcg'])
        # hit_loger.append(ret['hit_ratio'])

        # if args.verbose > 0:
        #     perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
        #                'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
        #                (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][-1],
        #                 ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
        #                 ret['ndcg'][0], ret['ndcg'][-1])
        #     print(perf_str)

        # cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
        #                                                             stopping_step, expected_order='acc', flag_step=5)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        # if should_stop == True:
        #     break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        # if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
        #     torch.save(model.state_dict(), args.weights_path + str(epoch) + '.pkl')
        #     print('save the weights in path: ', args.weights_path + str(epoch) + '.pkl')
    t2 = time()
    # ret = test(model, users_to_test, drop_flag=False)
    rate_matrix = test(model, users_to_test, n_items, drop_flag=False)
    t3 = time()
    output_file_name = "out_ngcf_e_" + str(args.epoch) + "_u_" + str(n_users//1000) + "k.csv"
    print("training finish and create output")
    main_process(playlists_tracks, test_playlists, train_playlists_count, args.batch_size, output_file_name)
    print("create output successflly!")
    # recs = np.array(rec_loger)
    # pres = np.array(pre_loger)
    # ndcgs = np.array(ndcg_loger)
    # hit = np.array(hit_loger)

    # best_rec_0 = max(recs[:, 0])
    # idx = list(recs[:, 0]).index(best_rec_0)

    # final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
    #              (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
    #               '\t'.join(['%.5f' % r for r in pres[idx]]),
    #               '\t'.join(['%.5f' % r for r in hit[idx]]),
    #               '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    # print(final_perf)