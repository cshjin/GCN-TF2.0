from scipy.sparse.csgraph import connected_components
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import networkx as nx
import numpy as np
import os
import pickle
import scipy.sparse as sp
import tensorflow as tf
from collections import defaultdict

def xavier_init(size):
    """ The initiation from the Xavier's paper
        ref: Understanding the difficulty of training deep feedforward neural 
            networks, Xavier Glorot, Yoshua Bengio, AISTATS 2010.
        http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

    Parameters
    ----------
    size: size of the variable

    Returns
    -------
    tf.tensor: an initialized variable

    """
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def sparse_dropout(x, dropout_rate, noise_shape):
    """ Dropout for sparse tensors

    Parameters
    ----------
    x: the tensor
    dropout_rate: probability of dropout
    noise_shape: shape of noise

    Returns
    -------
    tf.tensor: A tensor after dropout

    """
    random_tensor = 1-dropout_rate
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1./(1-dropout_rate))


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum.astype(float), -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.tocsr()


def load_data(dsname):
    """ Load splitted dataset from file, otherwise, dump the splited data to file

    Returns
    -------
    A_mat:
    X_mat:
    z_vec:
    train_idx:
    val_idx:
    test_idx:
    """
    filename = 'data/{}.pickle'.format(dsname)
    # if exists
    if os.path.exists(filename):
        print('>>> Loading data from file!!!')
        with open(filename, 'rb') as f:
            A_mat, X_mat, z_vec, train_idx, val_idx, test_idx = pickle.load(f)
            return A_mat, X_mat, z_vec, train_idx, val_idx, test_idx

    # if not exists
    A_mat, X_mat, z_vec = load_npz('data/{}.npz'.format(dsname))
    X_mat = X_mat.astype(np.float32)

    # remove non-connected component
    G = nx.from_scipy_sparse_matrix(A_mat)
    G = max(nx.connected_component_subgraphs(G), key=len)
    # remove selfloop edges
    G.remove_edges_from(G.selfloop_edges())
    A_mat = nx.adjacency_matrix(G)
    assert A_mat.nnz == len(G.edges) * 2
    degrees = A_mat.sum(0).A1
    nodes = G.nodes
    X_mat = X_mat[nodes]
    z_vec = z_vec[nodes]
    N = A_mat.shape[0]
    train_size = int(0.1 * N)
    val_size = int(0.1 * N)
    test_size = N - train_size - val_size
    idx = np.arange(N)
    train_idx, test_idx = train_test_split(idx,
                                           train_size=train_size + val_size,
                                           test_size=test_size,
                                           random_state=10)
    train_idx, val_idx = train_test_split(train_idx,
                                          train_size=train_size,
                                          test_size=val_size,
                                          random_state=10)

    with open(filename, 'wb') as f:
        print('>>> Dumping datasplit to file!!!')
        pickle.dump([A_mat, X_mat, z_vec, train_idx, val_idx, test_idx], f)
    return A_mat, X_mat, z_vec, train_idx, val_idx, test_idx


def load_data_planetoid(dataset):
    """ Load dataset of the splitted version from Planetoid  

    Parameters
    ----------
    dataset: string
        name of the dataset

    Returns
    -------
    A_mat,
    X_mat,
    z_vec,
    idx_train,
    idx_val,
    idx_test
    """
    if dataset not in ['citeseer', 'cora', 'pubmed']:
        print("No dataset found!")
    keys = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = defaultdict()
    for key in keys:
        with open('data_split/ind.{}.{}'.format(dataset, key), 'rb') as f:
            objects[key] = pickle.load(f, encoding='latin1')
    test_index = [int(x) for x in open('data_split/ind.{}.test.index'.format(dataset))]
    test_index_sort = np.sort(test_index)
    G = nx.from_dict_of_lists(objects['graph'])

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_index), max(test_index)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), objects['x'].shape[1]))
        tx_extended[test_index_sort-min(test_index_sort), :] = objects['tx']
        objects['tx'] = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), objects['y'].shape[1]))
        ty_extended[test_index_sort-min(test_index_sort), :] = objects['ty']
        objects['ty'] = ty_extended

    A_mat = nx.adjacency_matrix(G)
    X_mat = sp.vstack((objects['allx'], objects['tx'])).tolil()
    X_mat[test_index, :] = X_mat[test_index_sort, :]
    z_vec = np.vstack((objects['ally'], objects['ty']))
    z_vec[test_index, :] = z_vec[test_index_sort, :]
    z_vec = z_vec.argmax(1)

    train_idx = range(len(objects['y']))
    val_idx = range(len(objects['y']), len(objects['y'])+500)
    test_idx = test_index_sort.tolist()

    return A_mat, X_mat, z_vec, train_idx, val_idx, test_idx


def load_npz(file_name):
    """Load a SparseGraph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    sparse_graph : gust.SparseGraph
        Graph in sparse matrix format.

    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        # retrieve A
        assert 'adj_data' in loader
        adj_matrix = sp.csr_matrix((loader['adj_data'],
                                    loader['adj_indices'],
                                    loader['adj_indptr']),
                                   shape=loader['adj_shape'])

        # retrieve X
        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'],
                                         loader['attr_indices'],
                                         loader['attr_indptr']),
                                        shape=loader['attr_shape'])
        else:
            degrees = adj_matrix.sum(1).A1
            attr_matrix = sp.diags(degrees)
            attr_matrix = attr_matrix.tocsr()

        # retrieve z
        assert 'labels' in loader
        labels = loader.get('labels')

    return adj_matrix, attr_matrix, labels


def largest_connected_components(adj, n_components=1):
    """Select the largest connected components in the graph.

    Parameters
    ----------
    sparse_graph : gust.SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.

    Returns
    -------
    sparse_graph : gust.SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.

    """
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep


    ]
    # print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep


def train_val_test_split_tabular(*arrays, train_size=0.5, val_size=0.3, test_size=0.2, stratify=None, random_state=None):
    """
    Split the arrays or matrices into random train, validation and test subsets.

    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
            Allowed inputs are lists, numpy arrays or scipy-sparse matrices.
    train_size : float, default 0.5
        Proportion of the dataset included in the train split.
    val_size : float, default 0.3
        Proportion of the dataset included in the validation split.
    test_size : float, default 0.2
        Proportion of the dataset included in the test split.
    stratify : array-like or None, default None
        If not None, data is split in a stratified fashion, using this as the class labels.
    random_state : int or None, default None
        Random_state is the seed used by the random number generator;

    Returns
    -------
    splitting : list, length=3 * len(arrays)
        List containing train-validation-test split of inputs.

    """
    # DEBUG: fix the error when sum(train_size + test_size) != samples
    if len(set(array.shape[0] for array in arrays)) != 1:
        raise ValueError("Arrays must have equal first dimension.")
    idx = np.arange(arrays[0].shape[0])
    # train_sample = len(idx) * train_size
    # val_sample = len(idx) * val_size
    # test_sample = len(idx) - train_sample - val_sample
    idx_train_and_val, idx_test = train_test_split(
        idx,
        random_state=random_state,
        train_size=train_size + val_size,
        test_size=test_size,
        stratify=stratify)

    if stratify is not None:
        stratify = stratify[idx_train_and_val]
        idx_train, idx_val = train_test_split(
            idx_train_and_val,
            random_state=random_state,
            train_size=train_size,
            test_size=val_size,
            stratify=stratify)

    result = []
    for X in arrays:
        result.append(X[idx_train])
        result.append(X[idx_val])
        result.append(X[idx_test])
    return result


def preprocess_graph(adj, c=1):
    """ process the graph
        * options:
        normalization of augmented adjacency matrix
        formulation from convolutional filter
        normalized graph laplacian

    Parameters
    ----------
    adj: a sparse matrix represents the adjacency matrix

    Returns
    -------
    adj_normalized: a sparse matrix represents the normalized laplacian
        matrix
    """
    _adj = adj + c * sp.eye(adj.shape[0])
    _dseq = _adj.sum(1).A1
    _D = sp.diags(_dseq)
    _D_half = sp.diags(np.power(_dseq, -0.5))
    adj_normalized = _D_half @ _adj @ _D_half
    return adj_normalized.tocsr()


def preprocess_graph2(adj, c=1):
    """ process the graph
        * options:
        normalization of augmented adjacency matrix
        formulation from convolutional filter
        normalized graph laplacian

    Parameters
    ----------
    adj: a sparse matrix represents the adjacency matrix

    Returns
    -------
    An: a sparse matrix represents the normalized laplacian
        matrix
    """
    adj_orig = adj
    adj = adj + c * sp.eye(adj.shape[0])
    dseq = adj_orig.sum(1).A1
    D = sp.diags(dseq)
    D_inv = sp.diags(np.power(dseq, -1.))
    D_inv_sqrt = sp.diags(np.power(dseq, -0.5))
    An = D_inv @ adj
    return An.tocsr()


def correct_predicted(y_true, y_pred):
    """ Compare the ground truth and predict labels,

    Parameters
    ----------
    y_true: an array like for the true labels
    y_pred: an array like for the predicted labels

    Returns
    -------
    correct_predicted_idx: a list of index of correct predicted 
    correct_score: a rate of accuracy rate

    H. J. @ 2018-12-18
    """
    if len(y_true) != len(y_pred):
        raise "Dimension unmatches"
    correct_predicted_idx = []
    for idx in range(len(y_true)):
        if y_pred[idx] == y_true[idx]:
            correct_predicted_idx.append(idx)
    correct_score = accuracy_score(y_true, y_pred)

    return correct_predicted_idx, correct_score


def compute_margin_score(y_true, y_pred_prob, N=2):
    """ Implementation of the margin score
        Def: X = Z_{v, c_{old}} - \max_{c \neq c_{old}} Z_{v, c}
        Pick Top N and Last N nodes as the candidate nodes

    Parameters
    ----------
    y_true: an array like true labels
    y_pred_prob: an array like predicted probabilities
    N: number of candidate picked to attack

    Returns
    -------
    margin_scores: list of margin scores for each nodes
    picked_nodes: list of nodes picked from topN and lastN

        H. J. @ 2018-12-18
    """
    # size of y_true
    size = len(y_true)

    margin_score = []
    predict_labels = y_pred_prob.argmax(axis=1)
    for i in range(size):
        if y_true[i] == predict_labels[i]:
            # _score is positive, correctly predicted
            _score = y_pred_prob[i][y_true[i]] - sorted(y_pred_prob[i])[-2]
        else:
            # _score is negative, incorrectly predicted
            _score = y_pred_prob[i][y_true[i]] - y_pred_prob[i][predict_labels[i]]
        margin_score.append(_score)

    # pick the nodes based on the top N margin scores
    margin_score = np.array(margin_score)
    margin_score_nonneg = margin_score.copy()
    margin_score_nonneg[margin_score_nonneg < 0] = 0
    topN = sorted(range(len(margin_score_nonneg)), key=lambda i: margin_score_nonneg[i], reverse=True)[:N]
    lastN = sorted(range(len(margin_score_nonneg)),
                   key=lambda i: margin_score_nonneg[i] if margin_score_nonneg[i] > 0 else 100)[:N]
    picked_nodes = topN + lastN

    return margin_score, picked_nodes


def compute_margin_score_v2(y_true, y_pred_prob, y_correct_idx, N=2):
    """ Implementation of the margin score
        Def: X = Z_{v, c_{old}} - \max_{c \neq c_{old}} Z_{v, c}
        Pick Top N and Last N nodes as the candidate nodes

    Parameters
    ----------
    y_true: an array like true labels
    y_pred_prob: an array like predicted probabilities
    y_correct_idx: an array like correctly predicted indices
    N: number of candidate picked to attack

    Returns
    -------
    margin_scores: list of margin scores for each nodes
    picked_nodes: list of nodes picked from topN and lastN
        H. J. @ 2018-12-18
    """
    size = len(y_true)

    margin_score = []
    predict_labels = y_pred_prob.argmax(axis=1)
    for i in y_correct_idx:
        _score = y_pred_prob[i][y_true[i]] - sorted(y_pred_prob[i])[-2]
        margin_score.append(_score)

    # pick the nodes based on the top N margin scores
    margin_score = np.array(margin_score)
    topN = sorted(range(len(margin_score)), key=lambda i: margin_score[i], reverse=True)[:N]
    lastN = sorted(range(len(margin_score)), key=lambda i: margin_score[i] if margin_score[i] > 0 else 100)[:N]
    # lastN = sorted(range(len(margin_score)), key=lambda i: margin_score[i])[:N]
    picked_nodes = topN + lastN
    picked_nodes = list(set(picked_nodes))
    return margin_score, picked_nodes


def compute_margin_score_v3(y_true, y_pred_prob, y_correct_idx, node_correct_idx, N=2):
    """ Implementation of the margin score
        Def: X = Z_{v, c_{old}} - \max_{c \neq c_{old}} Z_{v, c}
        Pick Top N and Last N nodes as the candidate nodes

    Parameters
    ----------
    y_true: an array like true labels
    y_pred_prob: an array like predicted probabilities
    y_correct_idx: an array like correctly predicted indices
    node_correct_idx: an array like correct idx for the predicted nodes
    N: number of candidate picked to attack

    Returns
    -------
    margin_scores: list of margin scores for each nodes
    picked_nodes: list of nodes picked from topN and lastN

    H. J. @ 2019-01-16
    """
    margin_score = {}
    predict_labels = y_pred_prob.argmax(axis=1)
    for node_idx, pred_idx in zip(node_correct_idx, y_correct_idx):
        _score = y_pred_prob[pred_idx][y_true[pred_idx]] - sorted(y_pred_prob[pred_idx])[-2]
        if _score >= 0:
            margin_score[node_idx] = _score

    # pick the nodes based on the top N margin scores
    size = min(N, len(margin_score))
    picked_nodes = sorted(margin_score, key=margin_score.get, reverse=True)[:size]
    return margin_score, picked_nodes


def normalized_laplacian_spectrum(G):
    """ Compute the eigenvalues of normalized Laplacian

    Parameters
    ----------
    G: a networkx graph

    Returns
    -------
    numpy array: eigenvalues of normalized laplacian

    reference: networkx.linalg.spectrum.laplacian_spectrum

    H. J. @ 2019-02-07
    """
    from scipy.linalg import eigvalsh
    return eigvalsh(nx.normalized_laplacian_matrix(G).todense())


def sp_matrix_to_sp_tensor(M):
    """ Convert a sparse matrix to a SparseTensor

    Parameters
    ----------
    M: a scipy.sparse matrix

    Returns
    -------
    X: a tf.SparseTensor 

    Notes
    -----
    Also see tf.SparseTensor, scipy.sparse.csr_matrix

    H. J. @ 2019-02-12
    """
    if not isinstance(M, sp.csr.csr_matrix):
        M = M.tocsr()
    row, col = M.nonzero()
    X = tf.SparseTensor(np.mat([row, col]).T, M.data, M.shape)
    X = tf.cast(X, tf.float32)
    return X


def sparse_to_tuple(sparse_mx):
    """ 
    Copyright (c) Thomas Kipf 
    Repo: https://github.com/tkipf/gae
    """
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges(adj):
    """ 
    Copyright (c) Thomas Kipf 
    Repo: https://github.com/tkipf/gae
    """
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false
