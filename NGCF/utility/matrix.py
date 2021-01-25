import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import sys
from tqdm import tqdm

def sample(batch_size, n_users, exist_users, train_items, n_items):
  if batch_size <= n_users:
      users = rd.sample(exist_users, batch_size)
  else:
      users = [rd.choice(exist_users) for _ in range(batch_size)]

  def sample_pos_items_for_u(u, num):
      # sample num pos items for u-th user
      pos_items = train_items[u]
      n_pos_items = len(pos_items)
      pos_batch = []
      while True:
          if len(pos_batch) == num:
              break
          pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
          pos_i_id = pos_items[pos_id]

          if pos_i_id not in pos_batch:
              pos_batch.append(pos_i_id)
      return pos_batch

  def sample_neg_items_for_u(u, num):
      # sample num neg items for u-th user
      neg_items = []
      while True:
          if len(neg_items) == num:
              break
          neg_id = np.random.randint(low=0, high=n_items, size=1)[0]
          if neg_id not in train_items[u] and neg_id not in neg_items:
              neg_items.append(neg_id)
      return neg_items

  # def sample_neg_items_for_u_from_pools(u, num):
  #     neg_items = list(set(self.neg_pools[u]) - set(train_items[u]))
  #     return rd.sample(neg_items, num)

  pos_items, neg_items = [], []
  for u in users:
      pos_items += sample_pos_items_for_u(u, 1)
      neg_items += sample_neg_items_for_u(u, 1)

  return users, pos_items, neg_items

def get_adj_mat(n_users, n_items, R, path):  
    try:
      print("find play_adj file")
      t1 = time()
      norm_adj_mat = sp.load_npz(path + '/play_norm_adj_mat.npz')
      #print('already load adj matrix', adj_mat.shape, time() - t1)

    except Exception:
      print("not find play_adj file")
      norm_adj_mat = create_adj_mat(n_users, n_items, R, path)
      sp.save_npz(path + '/play_norm_adj_mat.npz', norm_adj_mat)
    return norm_adj_mat


def create_adj_mat(n_users, n_items, R, path):
    t1 = time()
    adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R = R.tolil()

    #adj_mat[:n_users, n_users:] = R
    # adj_mat[n_users:, :n_users] = R.T
    min_batch_load = 2500
    min_batch_v = 100000
    times = n_users//min_batch_load
    min_times = n_items//min_batch_v
    for i in tqdm(range(times)):
      u_start = i*min_batch_load
      u_end = (i+1)*min_batch_load
      for j in range(min_times):
        v_start = j*min_batch_v
        v_end = (j+1)*min_batch_v
        tmp_R = R[u_start:u_end, v_start:v_end]
        adj_mat[u_start:u_end, n_users+ v_start:n_users+v_end] = tmp_R
        adj_mat[n_users+ v_start:n_users+v_end, u_start:u_end] = tmp_R.T
        if j == min_times-1:
          tmp_j = n_items % min_batch_v
          tmp_R = R[u_start:u_end, v_end:v_end+tmp_j]
          adj_mat[u_start:u_end, n_users+v_end:n_users+v_end+tmp_j] = tmp_R
          adj_mat[n_users+v_end:n_users+v_end+tmp_j, u_start:u_end] = tmp_R.T

    adj_mat = adj_mat.todok()
    print('already create adjacency matrix', adj_mat.shape, time() - t1)

    t2 = time()

    def mean_adj_single(adj):
        # D^-1 * A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        # norm_adj = adj.dot(d_mat_inv)
        print('generate single-normalized adjacency matrix.')
        return norm_adj.tocoo()

    norm_adj_mat = mean_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))

    print('already normalize adjacency matrix', time() - t2)
    return norm_adj_mat.tocsr()
