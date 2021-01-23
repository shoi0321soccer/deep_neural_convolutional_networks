import numpy as np
import random as rd
import scipy.sparse as sp
from time import time

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
