from __future__ import print_function, division

import numpy as np
from sklearn.metrics import pairwise_distances


class Coreset_Greedy:
    def __init__(self, all_pts, num_labeldata):
        self.all_pts = np.array(all_pts)
        self.dset_size = len(all_pts)
        self.num_label = num_labeldata
        self.min_distances = None
        # reshape
        #self.all_pts = self.all_pts.reshape(-1,2)

        # self.first_time = True

    def update_dist(self, centers, reset_dist=False):
        if reset_dist:
            self.min_distances = None

        if len(centers) != 0:
            x = [self.all_pts[i] for i in centers]
            dist = pairwise_distances(self.all_pts, x, metric='euclidean')

            self.min_distances = np.min(dist, axis=1).reshape(-1,1)  # 이게 문제.
    
    def sample(self, labeled_index_list, sample_size):

        # already_selected : 이전에 이미 labeling 된 것. 
        self.already_selected = labeled_index_list # label index가 여기에 들어가고 
        self.update_dist(labeled_index_list, reset_dist=True)

        # epdb.set_trace()

        new_batch = []
        # pdb.set_trace()
        for _ in range(sample_size):
            if len(self.already_selected) == 0 and len(new_batch)==0 :
                # ind 가 unlabeled data에 있도록 설정 
                ind = np.random.choice(np.arange(self.dset_size - self.num_label)) + self.num_label
            else:
                ind = np.argmax(self.min_distances)
                 
            # assert ind not in already_selected
            l1 = list(self.already_selected.copy())  
            new_batch.append(ind)
            l2 = l1 + new_batch 
            self.update_dist(l2, reset_dist=False)
            
        
        max_distance = max(self.min_distances)
        print("Max distance from cluster : %0.2f" % max_distance)

        return new_batch, max_distance