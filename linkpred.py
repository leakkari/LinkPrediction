import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import sklearn
from node2vec import Node2Vec
from sklearn.model_selection import *
from sklearn.linear_model import *
import lightgbm as lgbm
from sklearn.svm import SVC
import sys, os
from tqdm import tqdm


class LinkPred():
    def get_unconnected_nodes(G, source, target):
        '''
        Return all unconnected node pairs in a graph
        '''

        # combine all entities in a list
        node_list = source + target

        # remove dups from list
        node_list = list(dict.fromkeys(node_list))

        # build adj matrix
        adj_G = nx.to_numpy_matrix(G, nodelist=node_list)

        # Find positions of the zeros
        unconnected_pairs = []
        offset = 0

        for i in (range(adj_G.shape[0])):
            for j in range(offset, adj_G.shape[1]):
                if i != j:
                    if adj_G[i, j] == 0:
                        unconnected_pairs.append([node_list[i], node_list[j]])

            offset = offset + 1

        # These node pairs will act as negative samples during training of the link prediction model
        node1_unlinked = [i[0] for i in unconnected_pairs]
        node2_unlinked = [i[1] for i in unconnected_pairs]

        return unconnected_pairs, node1_unlinked, node2_unlinked

    def get_removable_links(G, kg_df):
        '''
        Return links that can be removed without breaking the graph
        '''

        # Check if dropping a link results in splitting graph
        node_nb = len(G.nodes)
        kg_df_temp = kg_df.copy()

        removable_links = []

        for i in (kg_df.index.values):
            # remove pair and build a new graph
            G_temp = nx.from_pandas_edgelist(kg_df_temp.drop(index=i), "node_1", "node_2", create_using=nx.Graph)

            # check nb of connect comp >1 && node_nb is the same
            if (nx.number_connected_components(G_temp) > 1 and len(G_temp.nodes) == node_nb):
                removable_links.append(i)
                kg_df_temp = kg_df_temp.drop(index=i)

        return removable_links

    def predict(nb_dropped_links):

        pairs = []
        # Read data out-putted from text-file
        x, relations, y = np.genfromtxt('coreProteins.txt', dtype='str', unpack=True)

        for i in range(100):
            pairs.append([x[i], y[i]])
        pairs = np.array(pairs)
        relations = np.array(relations)

        # extract entities
        nodes1 = [i[0] for i in pairs]
        nodes2 = [i[1] for i in pairs]

        # Get graph and Data Frame
        kg_df = pd.DataFrame({'node_1': nodes1, 'node_2': nodes2})

        print("\n*** CREATE DATA FRAME WITH ALL NODE CONNECTIONS *** \n")
        print("\nALL NODE CONNECTIONS DATA FRAME : \n", kg_df.head())
        print("\nALL NODE CONNECTIONS DATA FRAME SHAPE : ", kg_df.shape)

        print("\n*** CREATE GRAPH WILL ALL NODE CONNECTIONS *** \n")
        # create an undirected-graph from a dataframe
        G = nx.from_pandas_edgelist(kg_df, "node_1", "node_2", create_using=nx.Graph())
        print("\nALL NODE CONNECTIONS Graph INFO :\n ", nx.info(G))

        # save graph
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G)
        nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos=pos)
        # plt.show()
        plt.savefig('./Undirected_Graph.png')

        # Get unconnected pairs
        unconnected_pairs, node1_unlinked, node2_unlinked = LinkPred.get_unconnected_nodes(G, nodes1, nodes2)

        # Build Data Frame
        data = pd.DataFrame({'node_1': node1_unlinked, 'node_2': node2_unlinked})
        data['link'] = 0

        print("\n*** CREATE DATA FRAME WITH ALL UNCONNECTED NODE PAIRS *** \n")
        print("\nALL NODE NO-CONNECTIONS MODEL DATA FRAME : \n", data.head())
        print("\nALL NODE NO-CONNECTIONS MODEL DATA FRAME SHAPE : ", data.shape)

        removable_links = LinkPred.get_removable_links(G, kg_df)
        print("\nLINKS THAT CAN BE REMOVED ARE : ", removable_links)
        # Append removable edges to dataframe of unconnected node pairs

        # Remove 3 connections
        links = random.sample(removable_links, nb_dropped_links)
        kg_df_ghost = kg_df.loc[links]
        kg_df_ghost['link'] = 1
        #
        # kg_df_ghost = kg_df.loc[removable_links[:nb_dropped_links]]


        print("\n*** THE FOLLOWING LINKS WERE REMOVED *** \n")
        print("\nRemoved Links : \n", kg_df_ghost.head())
        print("\nALL NODE REMOVABLE LINKS-GHOST DATA FRAME SHAPE : ", kg_df_ghost.shape)

        # data = data.append(kg_df_ghost[['node_1', 'node_2', 'link']], ignore_index=True)

        # Drop all removable links
        kg_df_partial = kg_df.drop(index=kg_df_ghost.index.values)
        kg_df_partial['link'] = 1


        # print("\n*DROP LINKS FROM ALL CONNECTIONS FAME* : \n", kg_df_partial.head())
        # print("\nALL NODE PARTIAL DATA FRAME : \n", kg_df_partial.head())
        # print("\nALL NODE PARTIAL DATA FRAME SHAPE : ", kg_df_partial.shape)

        data = data.append(kg_df_partial[['node_1', 'node_2', 'link']], ignore_index=True)
        print("\n*** THE DATA FRAME FOR TRAINING IS THE FOLLOWING: (ALL CONNECTED/UNCONNECTED PAIRS - DROPPED LINKS) ***")
        print("\nALL NODE FULL MODEL DATA FRAME : \n", data.head())
        print("\nALL NODE FULL MODEL DATA FRAME SHAPE : ", data.shape)



        # Make a new graph missing all the removable links
        G_data = nx.from_pandas_edgelist(kg_df_partial, "node_1", "node_2", create_using=nx.Graph)
        print("\nGRAPH AFTER DROPPING LINKS INFO :\n ", nx.info(G_data))
        print('\n')

        # Extract Features from graph

        node2vec = Node2Vec(G_data, dimensions=100, walk_length=16, num_walks=50)
        n2w_model = node2vec.fit(window=7, min_count=1)


        ######### TAKING DATA with 3 MISSING NODES INSTEAD of PARTIAL DATA
        x = [(n2w_model.wv.__getitem__(str(i)) + n2w_model.wv.__getitem__(str(j))) for i, j in zip(data['node_1'], data['node_2'])]
        x_ = [(n2w_model.wv.__getitem__(str(i)) + n2w_model.wv.__getitem__(str(j))) for i, j in zip(kg_df_ghost['node_1'], kg_df_ghost['node_2'])]

        # x_ = [(n2w_model[str(i)] + n2w_model[str(j)]) for i, j in zip(kg_df_ghost['node_1'], kg_df_ghost['node_2'])]


        x_train_3 = np.array(x)
        y_train_3 = data['link']

        x_test_3 = np.array(x_)
        y_test_3 = kg_df_ghost['link']

        lr = sklearn.linear_model.LogisticRegression(class_weight="balanced", max_iter=600)
        lr.fit(x_train_3, y_train_3)
        preds = lr.predict_proba(x_test_3)
        predictions = lr.predict(x_test_3)

        # print("\nACCURACY: ", acc_model)
        print("\n PAIRS   : \n", kg_df_ghost)
        print("\nEXPECTED : \n", np.array(y_test_3[:20]))
        print("\nRESULT   : \n", predictions[:20])
        score = sklearn.metrics.accuracy_score(y_test_3, predictions, normalize=True, sample_weight=None)

        return score

    # Disable
    def blockPrint(self):
        sys.stdout = open(os.devnull, 'w')

    # Restore
    def enablePrint(self):
        sys.stdout = sys.__stdout__


pass

if __name__ == '__main__':
   test = LinkPred()
   print("\n***************************** Link Prediction STARTED *****************************")
   score = LinkPred.predict(3)
   print("\n***************************** Link Prediction DONE ********************************")

   print("\n***************************** Compute Accuracy STARTED ****************************")


   accuracy = []
   MAX_ITER = 100


   for i in (range(MAX_ITER)):
       print("\nITERATION ", i+1, "/", MAX_ITER)
       test.blockPrint()
       score = LinkPred.predict(3)
       accuracy.append(score)
       test.enablePrint()

   accuracy = np.array(accuracy)
   avg_acc = np.mean(accuracy)


   test.enablePrint()
   print("\n***************************** Compute Accuracy DONE *****************************")
   print("\nOn average, the accuracy is: ", avg_acc)
