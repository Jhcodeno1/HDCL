import dgl
import torch
import numpy as np
import torch_sparse
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv, HGTConv, GraphConv
from sklearn.cluster import KMeans


# Semantic attention in the metapath-based aggregation (the same as that in the HAN)
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        '''
        Shape of z: (N, M , D*K)
        N: number of nodes
        M: number of metapath patterns
        D: hidden_size
        K: number of heads
        '''
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        return (beta * z).sum(1)  # (N, D * K)


# Metapath-based aggregation (the same as the HANLayer)
class HANLayer(nn.Module):
    def __init__(self, meta_path_patterns, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_path_patterns)):
            self.gat_layers.append(
                # GCN_layer()
                GraphConv(in_size, out_size, norm='both', weight=None, bias=None,
                          activation=None, allow_zero_in_degree=True)
            )
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_path_patterns = list(tuple(meta_path_pattern) for meta_path_pattern in meta_path_patterns)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []
        # obtain metapath reachable graph
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path_pattern in self.meta_path_patterns:
                self._cached_coalesced_graph[meta_path_pattern] = dgl.metapath_reachable_graph(
                    g, meta_path_pattern)

        for i, meta_path_pattern in enumerate(self.meta_path_patterns):
            new_g = self._cached_coalesced_graph[meta_path_pattern]
            # new_g = dgl.to_homogeneous(new_g)
            # coo = new_g.adj(scipy_fmt='coo', etype='_E')
            # csr_matrix = coo.tocsr()
            # semantic_embeddings.append(self.gat_layers[i](h, csr_matrix).flatten(1))
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class LightGCN(nn.Module):
    def __init__(self, g, args):
        super(LightGCN, self).__init__()
        self.g = g

        self.userkey = userkey = args.user_key  # 用户键
        self.itemkey = itemkey = args.item_key  # 物品键
        self.n_users = self.g.num_nodes(userkey)  # 用户数量
        self.n_items = self.g.num_nodes(itemkey)  # 物品数量
        n_nodes = self.n_users + self.n_items  # 总节点数

        # 构建邻接矩阵（未修改）
        row_idx = []
        col_idx = []
        adj = g.adj_external(ctx='cpu', scipy_fmt='csr', etype=args.ui_relation)
        for i in range(adj.shape[0]):
            start = adj.indptr[i]
            end = adj.indptr[i + 1]
            cols = adj.indices[start:end]
            for col in cols:
                row_idx.append(i)
                col_idx.append(col)
        col_idx = [idx + self.n_users for idx in col_idx]
        row_np = np.array(row_idx, dtype=np.int32)
        col_np = np.array(col_idx, dtype=np.int32)
        ratings = np.ones_like(row_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (row_np, col_np)), shape=(n_nodes, n_nodes), dtype=np.float32)
        self.ui_adj = tmp_adj + tmp_adj.T
        self.plain_adj = self.ui_adj
        rows, cols = self.ui_adj.nonzero()
        self.all_h_list = rows
        self.all_t_list = cols
        self.A_in_shape = self.plain_adj.tocoo().shape
        self.A_indices = torch.tensor([self.all_h_list, self.all_t_list], dtype=torch.long).cuda()
        self.D_indices = torch.tensor(
            [list(range(self.n_users + self.n_items)), list(range(self.n_users + self.n_items))],
            dtype=torch.long).cuda()
        self.all_h_list = torch.LongTensor(self.all_h_list).cuda()
        self.all_t_list = torch.LongTensor(self.all_t_list).cuda()
        self.G_indices, self.G_values = self._cal_sparse_adj()

        # 模型参数
        self.emb_dim = args.in_size  # 嵌入维度
        self.n_layers = 1  # GNN层数
        self.n_intents = 128  # 意图数量
        self.temp = 1  # 温度参数
        self.batch_size = args.batch_size  # 批次大小
        self.emb_reg = 2.5e-5  # 嵌入正则化系数
        self.cen_reg = 5e-3  # 中心正则化系数
        self.ssl_reg = 1e-1  # 自监督正则化系数

        # 0层扰动参数
        self.epsilon = args.epsilon if hasattr(args, 'epsilon') else 0.1  # 噪声幅度，可调超参数
        self.initial_embeddings = nn.Parameter(torch.empty(n_nodes, self.emb_dim))  # 初始嵌入
        nn.init.xavier_normal_(self.initial_embeddings)  # 使用Xavier正态分布初始化嵌入

    def _cal_sparse_adj(self):
        # 计算稀疏邻接矩阵（未修改）
        A_values = torch.ones(size=(len(self.all_h_list), 1)).view(-1).cuda()
        A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list, value=A_values,
                                             sparse_sizes=self.A_in_shape).cuda()
        D_values = A_tensor.sum(dim=1).pow(-0.5)
        G_indices, G_values = torch_sparse.spspmm(self.D_indices, D_values, self.A_indices, A_values,
                                                  self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])
        G_indices, G_values = torch_sparse.spspmm(G_indices, G_values, self.D_indices, D_values, self.A_in_shape[0],
                                                  self.A_in_shape[1], self.A_in_shape[1])
        return G_indices, G_values

    def _generate_perturbed_embeddings(self, embeddings):
        """使用0层扰动生成两个增强视图的嵌入。"""
        # 从均匀分布 U(0,1) 生成随机噪声
        delta1 = torch.rand_like(embeddings, device=embeddings.device)
        delta2 = torch.rand_like(embeddings, device=embeddings.device)

        # 保留符号并按 epsilon 缩放噪声
        delta1 = delta1 * torch.sign(embeddings) * self.epsilon
        delta2 = delta2 * torch.sign(embeddings) * self.epsilon

        # 生成扰动嵌入
        perturbed_emb1 = embeddings + delta1
        perturbed_emb2 = embeddings + delta2

        return perturbed_emb1, perturbed_emb2

    def forward(self, feature_dict):
        self.feature_dict = feature_dict
        base_embeddings = self.initial_embeddings

        # 使用0层扰动生成两个增强视图
        perturbed_emb1, perturbed_emb2 = self._generate_perturbed_embeddings(base_embeddings)

        # 通过GNN层处理原始嵌入和扰动嵌入
        all_embeddings = [base_embeddings]
        all_perturbed_emb1 = [perturbed_emb1]
        all_perturbed_emb2 = [perturbed_emb2]
        gnn_embeddings = []

        for i in range(self.n_layers):
            # 原始嵌入的图消息传递
            gnn_layer_embeddings = torch_sparse.spmm(self.G_indices, self.G_values, self.A_in_shape[0],
                                                     self.A_in_shape[1], all_embeddings[i])
            # 增强视图1的图消息传递
            gnn_layer_perturbed1 = torch_sparse.spmm(self.G_indices, self.G_values, self.A_in_shape[0],
                                                     self.A_in_shape[1], all_perturbed_emb1[i])
            # 增强视图2的图消息传递
            gnn_layer_perturbed2 = torch_sparse.spmm(self.G_indices, self.G_values, self.A_in_shape[0],
                                                     self.A_in_shape[1], all_perturbed_emb2[i])

            gnn_embeddings.append(gnn_layer_embeddings)
            all_embeddings.append(gnn_layer_embeddings + all_embeddings[i])
            all_perturbed_emb1.append(gnn_layer_perturbed1 + all_perturbed_emb1[i])
            all_perturbed_emb2.append(gnn_layer_perturbed2 + all_perturbed_emb2[i])

        # 跨层聚合嵌入
        all_embeddings = torch.stack(all_embeddings, dim=1).sum(dim=1, keepdim=False)
        all_perturbed_emb1 = torch.stack(all_perturbed_emb1, dim=1).sum(dim=1, keepdim=False)
        all_perturbed_emb2 = torch.stack(all_perturbed_emb2, dim=1).sum(dim=1, keepdim=False)

        # 分割为用户和物品嵌入
        ua_embedding, ia_embedding = torch.split(all_embeddings, [self.n_users, self.n_items], 0)
        ua_perturbed1, ia_perturbed1 = torch.split(all_perturbed_emb1, [self.n_users, self.n_items], 0)
        ua_perturbed2, ia_perturbed2 = torch.split(all_perturbed_emb2, [self.n_users, self.n_items], 0)

        return (ua_embedding, ia_embedding,
                ua_perturbed1, ia_perturbed1,
                ua_perturbed2, ia_perturbed2,
                gnn_embeddings)


class ComputeSimilarity:
    def __init__(self, model, dataMatrix, topk=10, shrink=0, normalize=True):
        r"""Computes the cosine similarity of dataMatrix

        If it is computed on :math:`URM=|users| \times |items|`, pass the URM.

        If it is computed on :math:`ICM=|items| \times |features|`, pass the ICM transposed.

        Args:
            dataMatrix (scipy.sparse.csr_matrix): The sparse data matrix.
            topk (int) : The k value in KNN.
            shrink (int) :  hyper-parameter in calculate cosine distance.
            normalize (bool):   If True divide the dot product by the product of the norms.
        """

        super(ComputeSimilarity, self).__init__()

        self.shrink = shrink
        self.normalize = normalize

        self.n_rows, self.n_columns = dataMatrix.shape
        topk = eval(topk)[0]
        self.TopK = min(topk, self.n_columns)

        self.dataMatrix = dataMatrix.copy()

        self.model = model

    def compute_similarity(self, method, block_size=100):
        r"""Compute the similarity for the given dataset

        Args:
            method (str) : Caculate the similarity of users if method is 'user', otherwise, calculate the similarity of items.
            block_size (int): divide matrix to :math:`n\_rows \div block\_size` to calculate cosine_distance if method is 'user',
                 otherwise, divide matrix to :math:`n\_columns \div block\_size`.

        Returns:

            list: The similar nodes, if method is 'user', the shape is [number of users, neigh_num],
            else, the shape is [number of items, neigh_num].
            scipy.sparse.csr_matrix: sparse matrix W, if method is 'user', the shape is [self.n_rows, self.n_rows],
            else, the shape is [self.n_columns, self.n_columns].
        """

        user_similar_neighbors_mat, item_similar_neighbors_mat = [], []
        user_similar_neighbors_weights_mat, item_similar_neighbors_weights_mat = [], []

        values = []
        rows = []
        cols = []
        neigh = []

        self.dataMatrix = self.dataMatrix.astype(np.float32)

        # Compute sum of squared values to be used in normalization
        if method == 'user':
            sumOfSquared = np.array(self.dataMatrix.power(2).sum(axis=1)).ravel()
            end_local = self.n_rows
        elif method == 'item':
            sumOfSquared = np.array(self.dataMatrix.power(2).sum(axis=0)).ravel()
            end_local = self.n_columns
        else:
            raise NotImplementedError("Make sure 'method' in ['user', 'item']!")
        sumOfSquared = np.sqrt(sumOfSquared)

        start_block = 0

        # Compute all similarities using vectorization
        while start_block < end_local:

            end_block = min(start_block + block_size, end_local)
            this_block_size = end_block - start_block

            # All data points for a given user or item
            if method == 'user':
                data = self.dataMatrix[start_block:end_block, :]
            else:
                data = self.dataMatrix[:, start_block:end_block]
            data = data.toarray().squeeze()

            if data.ndim == 1:  # 如果 data 是一维数组，将其扩展为二维数组
                data = np.expand_dims(data, axis=1)

            # Compute similarities

            if method == 'user':
                this_block_weights = self.dataMatrix.dot(data.T)
            else:
                this_block_weights = self.dataMatrix.T.dot(data)

            for index_in_block in range(this_block_size):

                if this_block_size == 1:
                    this_line_weights = this_block_weights.squeeze()
                else:  # 提取当前 index_in_block 的相似度
                    this_line_weights = this_block_weights[:, index_in_block]

                Index = index_in_block + start_block
                this_line_weights[Index] = 0.0  # 设置自身相似度为0：防止用户或物品与自己进行比较

                # Apply normalization and shrinkage, ensure denominator != 0
                if self.normalize:
                    denominator = sumOfSquared[Index] * sumOfSquared + self.shrink + 1e-6
                    this_line_weights = np.multiply(this_line_weights, 1 / denominator)

                elif self.shrink != 0:
                    this_line_weights = this_line_weights / self.shrink

                # Sort indices and select TopK
                # Sorting is done in three steps. Faster then plain np.argsort for higher number of users or items
                # - Partition the data to extract the set of relevant users or items
                # - Sort only the relevant users or items
                # - Get the original index
                # argpartition 可以快速找到前 TopK 个最大相似度值的索引，但这些索引对应的值并不是完全排序的，只是保证了这 TopK 个值是最大的
                relevant_partition = (-this_line_weights).argpartition(self.TopK - 1)[0:self.TopK]
                # argsort 对 argpartition 找到的前 TopK 个值进行降序排序，确保我们得到的是从大到小排列的前 TopK 个相似度值的索引
                relevant_partition_sorting = np.argsort(-this_line_weights[relevant_partition])
                top_k_idx = relevant_partition[relevant_partition_sorting]
                neigh.append(top_k_idx)

                # Incrementally build sparse matrix, do not add zeros
                notZerosMask = this_line_weights[top_k_idx] != 0.0  # 把 0 值去掉
                tmp_values = this_line_weights[top_k_idx][notZerosMask]

                if method == 'user':
                    user_similar_neighbors_mat.append(top_k_idx[notZerosMask])  # 具体的最近邻居的id, 把 0 值去掉了
                    user_similar_neighbors_weights_mat.append(tmp_values)  # 具体的最近邻居的相似度值
                else:
                    item_similar_neighbors_mat.append(top_k_idx[notZerosMask])
                    item_similar_neighbors_weights_mat.append(tmp_values)

            start_block += block_size

        if method == 'user':
            return user_similar_neighbors_mat, user_similar_neighbors_weights_mat
        elif method == 'item':
            return item_similar_neighbors_mat, item_similar_neighbors_weights_mat


class HDCL(nn.Module):
    def __init__(self, g, args):
        super(HDCL, self).__init__()
        self.g = g
        self.user_key = user_key = args.user_key
        self.item_key = item_key = args.item_key
        self.unum = self.n_users = self.g.num_nodes(user_key)
        self.inum = self.n_items = self.g.num_nodes(item_key)
        self.device = args.device
        self.han_layers = 1

        self.initializer = nn.init.xavier_uniform_
        self.feature_dict = nn.ParameterDict({
            ntype: nn.Parameter(self.initializer(torch.empty(g.num_nodes(ntype), args.in_size))) for ntype in g.ntypes
        })
        self.LightGCN = LightGCN(g, args)
        # metapath-based aggregation modules for user and item, this produces h2
        self.meta_path_patterns = args.meta_path_patterns
        # one HANLayer for user, one HANLayer for item
        self.hans = nn.ModuleDict({
            key: HANLayer(value, args.in_size, args.out_size, args.num_heads, args.dropout) for key, value in
            self.meta_path_patterns.items()
        })

        # Cluster Anchor Regularization Setup
        self.num_clusters = args.num_clusters  # e.g., 768 as in the paper
        self.cluster_level = args.cluster_level  # e.g., 'level-2'
        self.lambda_H = args.lambda_H  # e.g., 0.01 hot project
        self.lambda_T = args.lambda_T  # e.g., 1.0 tail
        # 多层次锚点和簇分配
        self.cluster_anchors = nn.ParameterDict()
        self.cluster_labels = {'user': [], 'item': []}
        self._compute_multi_level_clusters(args)
        self.head_persent=args.head_persent


        self.ssl_temp = 0.1
        self.cl_rate = args.cl_rate
        self.ts=args.ts

        self.user_similar_neighbors_mat, self.user_similar_neighbors_weights_mat, \
            self.item_similar_neighbors_mat, self.item_similar_neighbors_weights_mat = self.get_similar_users_items(
            args)

        self.gamma = args.gamma
        self.beta = args.beta

    def _compute_multi_level_clusters(self, args):
        """计算多层次聚类"""
        with torch.no_grad():
            user_emb = self.feature_dict[self.user_key].cpu().numpy()
            item_emb = self.feature_dict[self.item_key].cpu().numpy()

            # 层次聚类
            for level in range(self.cluster_level):
                num_clusters = self.num_clusters // (2 ** level)  # 每层簇数减半
                if num_clusters < 1:
                    num_clusters = 1  # 保证至少 1 个簇

                # 用户聚类
                if level == 0:
                    kmeans_users = KMeans(n_clusters=num_clusters, random_state=42).fit(user_emb)
                    self.cluster_labels['user'].append(torch.tensor(kmeans_users.labels_, device=self.device))
                else:
                    prev_labels = self.cluster_labels['user'][level - 1].cpu().numpy()
                    level_labels = np.zeros_like(prev_labels)
                    for cluster_idx in range(self.cluster_labels['user'][level - 1].max() + 1):
                        mask = prev_labels == cluster_idx
                        if mask.sum() > 1:  # 确保有足够样本
                            sub_emb = user_emb[mask]
                            sub_kmeans = KMeans(n_clusters=min(num_clusters, mask.sum()), random_state=42).fit(sub_emb)
                            level_labels[mask] = sub_kmeans.labels_
                    self.cluster_labels['user'].append(torch.tensor(level_labels, device=self.device))

                # 物品聚类
                if level == 0:
                    kmeans_items = KMeans(n_clusters=num_clusters, random_state=42).fit(item_emb)
                    self.cluster_labels['item'].append(torch.tensor(kmeans_items.labels_, device=self.device))
                else:
                    prev_labels = self.cluster_labels['item'][level - 1].cpu().numpy()
                    level_labels = np.zeros_like(prev_labels)
                    for cluster_idx in range(self.cluster_labels['item'][level - 1].max() + 1):
                        mask = prev_labels == cluster_idx
                        if mask.sum() > 1:
                            sub_emb = item_emb[mask]
                            sub_kmeans = KMeans(n_clusters=min(num_clusters, mask.sum()), random_state=42).fit(sub_emb)
                            level_labels[mask] = sub_kmeans.labels_
                    self.cluster_labels['item'].append(torch.tensor(level_labels, device=self.device))

                # 初始化锚点
                self.cluster_anchors[f'user_level_{level}'] = nn.Parameter(
                    self.initializer(torch.empty(num_clusters, args.in_size)))
                self.cluster_anchors[f'item_level_{level}'] = nn.Parameter(
                    self.initializer(torch.empty(num_clusters, args.in_size)))

    def _cluster_anchor_regularization(self, ua_embedding, ia_embedding, h2):
        """计算多层次 CAR 损失"""
        user_emb = 0.5 * ua_embedding + 0.5 * h2[self.user_key]
        item_emb = 0.5 * ia_embedding + 0.5 * h2[self.item_key]

        # 动态头部和尾部划分
        interaction_counts = self.interaction_matrix.sum(axis=1).A1
        user_head_mask = interaction_counts >= np.percentile(interaction_counts, self.head_persent)
        item_head_mask = self.interaction_matrix.sum(axis=0).A1 >= np.percentile(self.interaction_matrix.sum(axis=0).A1,
                                                                                 self.head_persent)

        user_head_idx = torch.tensor(np.where(user_head_mask)[0], device=self.device)
        user_tail_idx = torch.tensor(np.where(~user_head_mask)[0], device=self.device)
        item_head_idx = torch.tensor(np.where(item_head_mask)[0], device=self.device)
        item_tail_idx = torch.tensor(np.where(~item_head_mask)[0], device=self.device)

        total_car_loss = 0
        for level in range(self.cluster_level):
            user_clusters = self.cluster_labels['user'][level]
            item_clusters = self.cluster_labels['item'][level]

            # 源正则化 (L_S)
            L_S_user = torch.mean((self.cluster_anchors[f'user_level_{level}'][user_clusters[user_head_idx]] - user_emb[
                user_head_idx].detach()) ** 2) if len(user_head_idx) > 0 else 0
            L_S_item = torch.mean((self.cluster_anchors[f'item_level_{level}'][item_clusters[item_head_idx]] - item_emb[
                item_head_idx].detach()) ** 2) if len(item_head_idx) > 0 else 0
            L_S = (L_S_user + L_S_item) / 2.0 if (L_S_user > 0 or L_S_item > 0) else torch.tensor(0.0,
                                                                                                  device=self.device)

            # 目标正则化 (L_T)
            L_T_user = torch.mean((user_emb[user_tail_idx] - self.cluster_anchors[f'user_level_{level}'][
                user_clusters[user_tail_idx]].detach()) ** 2) if len(user_tail_idx) > 0 else 0
            L_T_item = torch.mean((item_emb[item_tail_idx] - self.cluster_anchors[f'item_level_{level}'][
                item_clusters[item_tail_idx]].detach()) ** 2) if len(item_tail_idx) > 0 else 0
            L_T = (L_T_user + L_T_item) / 2.0 if (L_T_user > 0 or L_T_item > 0) else torch.tensor(0.0,
                                                                                                  device=self.device)

            # 每层损失加权（越高层权重越低）
            level_weight = 1.0 / (level + 1)
            total_car_loss += level_weight * (self.lambda_H * L_S + self.lambda_T * L_T)

        return total_car_loss

    def get_similar_users_items(self, args):
        # load parameters info
        self.k = args.topK
        self.shrink = args['shrink'] if 'shrink' in args else 0.0  # 调节相似度计算的结果

        row_idx = []
        col_idx = []
        adj = self.g.adj_external(ctx='cpu', scipy_fmt='csr', etype=args.ui_relation)
        for i in range(adj.shape[0]):  # 每一行代表 与目标类型id=i相连的srcType的节点ID
            # 从 CSC 矩阵中获取第 i 列的所有非零行索引，即所有与列索引 i 有连接的节点
            start = adj.indptr[i]
            end = adj.indptr[i + 1]
            cols = adj.indices[start:end]
            # 记录行索引和列索引
            for col in cols:
                row_idx.append(i)
                col_idx.append(col)
        # 将列索引转换成物品索引，确保它们在用户索引之后
        # col_idx = [idx + self.n_users for idx in col_idx]
        # 转换为 NumPy 数组
        row_np = np.array(row_idx, dtype=np.int32)
        col_np = np.array(col_idx, dtype=np.int32)
        # 创建一个与 user_np 相同长度的全 1 数组
        ratings = np.ones_like(row_np, dtype=np.float32)
        # 构建新的稀疏矩阵
        tmp_adj = sp.csr_matrix((ratings, (row_np, col_np)), shape=(self.unum, self.inum), dtype=np.float32)
        self.interaction_matrix = interaction_matrix = tmp_adj
        # interaction_matrix = dataset.inter_matrix(form='csr').astype(np.float32)
        shape = interaction_matrix.shape
        assert self.n_users == shape[0] and self.n_items == shape[1]

        user_similar_neighbors_mat, user_similar_neighbors_weights_mat = ComputeSimilarity(self, interaction_matrix,
                                                                                           topk=self.k,
                                                                                           shrink=self.shrink).compute_similarity(
            'user')

        item_similar_neighbors_mat, item_similar_neighbors_weights_mat = ComputeSimilarity(self, interaction_matrix,
                                                                                           topk=self.k,
                                                                                           shrink=self.shrink).compute_similarity(
            'item')

        return user_similar_neighbors_mat, user_similar_neighbors_weights_mat, item_similar_neighbors_mat, item_similar_neighbors_weights_mat

    def ssl_loss(self, data1, data2, index):
        index = torch.unique(index)
        embeddings1 = data1[index]
        embeddings2 = data2[index]
        norm_embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        norm_embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        pos_score = torch.sum(torch.mul(norm_embeddings1, norm_embeddings2), dim=1)
        all_score = torch.mm(norm_embeddings1, norm_embeddings2.T)
        pos_score = torch.exp(pos_score / 0.5)
        all_score = torch.sum(torch.exp(all_score / 0.5), dim=1)
        ssl_loss = (-torch.sum(torch.log(pos_score / ((all_score)))) / (len(index)))
        return ssl_loss

    def neighbor_sample(self, input_list, weight_list):
        if len(input_list) == 1:  # 只有一个邻居可选，不需要进行采样
            return input_list[0], weight_list[0]
        else:
            self.prob_sampling = True
            if self.prob_sampling:  # 使用概率抽样
                prob = np.asarray(weight_list).astype('float64')
                prob = prob / sum(prob)
                idx = np.random.choice(range(0, len(input_list)), size=1, replace=True, p=prob)
                idx = idx.item()
            else:
                idx = np.random.randint(0, len(input_list))
            return input_list[idx], weight_list[idx]

    def calculate_ssl_loss(self, data1, data2, user, pos_item):
        batch_user_weight = []
        batch_item_weight = []

        # batch_users_3 is used to index the user embedding from view-2
        batch_users_3 = []
        # batch_items_3 is used to index the user embedding from view-2
        batch_items_3 = []

        # batch_users_4 is used to index the user embedding from view-1
        batch_users_4 = []
        # batch_items_4 is used to index the user embedding from view-1
        batch_items_4 = []

        batch_nodes_list = []

        with torch.no_grad():
            batch_users_list = user.cpu().numpy().tolist()
            # update item ids to map the original item id to the constructed graph
            batch_items_list = (pos_item + self.unum).cpu().numpy().tolist()

            # batch_nodes_list stores both the batch_users_list and the batch_item_list
            batch_nodes_list.extend(batch_users_list)
            batch_nodes_list.extend(batch_items_list)

            for idx, user in enumerate(batch_users_list):
                # 两个视图的相同节点视为正样本
                batch_user_weight.append(1.0)  # 1.0是权重值
                batch_users_3.append(user)
                batch_users_4.append(user)

                # add user-item positive pair
                item = batch_items_list[idx]
                batch_user_weight.append(1.0)
                batch_users_3.append(item)  # 加了一个训练的和user配对的正样本pos_item
                batch_users_4.append(user)

                # add user and her k-nearest neighbors positive pair
                if self.user_similar_neighbors_mat[user].size != 0:  # 如果有可选的候选最近邻邻居
                    sample_user, sample_weight = self.neighbor_sample(self.user_similar_neighbors_mat[user],
                                                                      self.user_similar_neighbors_weights_mat[
                                                                          user])
                    batch_user_weight.append(1.0)  # 1.0
                    batch_users_3.append(sample_user)  # 从中挑选一个最近邻居
                    batch_users_4.append(user)

            for idx, item in enumerate(batch_items_list):
                batch_item_weight.append(1.0)
                batch_items_3.append(item)
                batch_items_4.append(item)

                # add item-user positive pair
                user = batch_users_list[idx]
                batch_item_weight.append(1.0)
                batch_items_3.append(user)
                batch_items_4.append(item)

                # # add item and its k-nearest neighbors positive pair
                if self.item_similar_neighbors_mat[item - self.n_users].size != 0:
                    sample_item, sample_weight = self.neighbor_sample(
                        self.item_similar_neighbors_mat[item - self.n_users],
                        self.item_similar_neighbors_weights_mat[item - self.n_users])
                    sample_item += +self.n_users
                    batch_item_weight.append(1.0)
                    batch_items_3.append(sample_item)
                    batch_items_4.append(item)

            batch_users_3 = torch.tensor(batch_users_3).long().to(self.device)
            batch_items_3 = torch.tensor(batch_items_3).long().to(self.device)
            batch_users_4 = torch.tensor(batch_users_4).long().to(self.device)
            batch_items_4 = torch.tensor(batch_items_4).long().to(self.device)
            batch_nodes_list = torch.tensor(list(batch_nodes_list)).long().to(self.device)

        # batch_users_3, batch_items_3 are consisf of different positive samples, get representations from view-1
        user_emb3 = data1[batch_users_3]
        item_emb3 = data2[batch_items_3]

        # batch_users_4, batch_items_4 are consisf of the anchor nodes themseleves, get representations from view-2
        user_emb4 = data1[batch_users_4]
        item_emb4 = data2[batch_items_4]

        # get representations from view-2
        batch_node_emb = data2[batch_nodes_list]

        emb_merge3 = torch.cat([user_emb3, item_emb3], dim=0)
        emb_merge4 = torch.cat([user_emb4, item_emb4], dim=0)

        batch_weights = batch_user_weight + batch_item_weight
        batch_weights = torch.tensor(batch_weights, dtype=torch.float32).to(self.device)

        # cosine similarity
        normalize_emb_merge3 = torch.nn.functional.normalize(emb_merge3, p=2, dim=1)
        normalize_emb_merge4 = torch.nn.functional.normalize(emb_merge4, p=2, dim=1)
        normalize_batch_node_emb = torch.nn.functional.normalize(batch_node_emb, p=2, dim=1)

        # differeent kinds of positive samples from view-1 mutliply the anchor nodes' representations from view-2
        pos_score = torch.sum(torch.multiply(normalize_emb_merge3, normalize_emb_merge4), dim=1)

        # different kinds of positive samples from view-1 matmul the negative samples from view-2
        ttl_score = torch.matmul(normalize_emb_merge3, normalize_batch_node_emb.transpose(0, 1))

        pos_score = batch_weights * torch.exp(pos_score / self.ssl_temp)
        ttl_score = torch.sum(torch.exp(ttl_score / self.ssl_temp), dim=1)
        ssl_loss = -torch.mean(torch.log(pos_score / ttl_score))

        return ssl_loss

    def alignment(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def uniformity(self, x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def calculate_loss(self, user_e, item_e):
        # user_e, item_e = self.encoder(user, item)  # [bsz, dim]
        align = self.alignment(user_e, item_e)
        uniform = (self.uniformity(user_e) + self.uniformity(item_e)) / 2
        return align, uniform

    def forward(self, user_idx, item_idx, neg_item_idx):
        ua_embedding, ia_embedding, ua_embedding1, ia_embedding1,ua_embedding2, ia_embedding2,int_embeddings = self.LightGCN(self.feature_dict)
        # metapath-based aggregation, h2
        h2 = {}
        for key in self.meta_path_patterns.keys():
            for i in range(self.han_layers):
                if i == 0:
                    h2[key] = self.hans[key](self.g, self.feature_dict[key])
                else:
                    h2[key] = self.hans[key](self.g, h2[key])
        user_emb = 0.5 * ua_embedding + 0.5 * h2[self.user_key]
        item_emb = 0.5 * ia_embedding + 0.5 * h2[self.item_key]

        car_loss = self._cluster_anchor_regularization(ua_embedding, ia_embedding, h2)
        # 计算对比损失
        # ssl_loss_user = self.ssl_loss(ua_embedding1, ua_embedding2, user_idx)
        # ssl_loss_item = self.ssl_loss(ia_embedding1, ia_embedding2, item_idx)
        # ssl_loss = ssl_loss_user + ssl_loss_item
        data1 = torch.cat((h2[self.user_key], h2[self.item_key]), dim=0)
        data2 = torch.cat((ua_embedding, ia_embedding), dim=0)
        ssl_loss = self.calculate_ssl_loss(data1, data2, user_idx, item_idx)

        user_e3 = ua_embedding1[user_idx]
        item_e3 = ia_embedding1[item_idx]

        user_e4 = ua_embedding2[user_idx]
        item_e4 = ia_embedding2[item_idx]

        user_e1 = ua_embedding[user_idx]
        item_e1 = ia_embedding[item_idx]

        user_e2 = h2[self.user_key][user_idx]
        item_e2 = h2[self.item_key][item_idx]

        align_loss_item, unif_loss_item = self.calculate_loss(item_e1, item_e2)
        align_loss_user, unif_loss_user = self.calculate_loss(user_e1, user_e2)
        align_loss = (align_loss_user + align_loss_item) / 2
        uniform_loss = (unif_loss_user + unif_loss_item) / 2
        align_loss_item, unif_loss_item = self.calculate_loss(item_e3, item_e4)
        align_loss_user, unif_loss_user = self.calculate_loss(user_e3,user_e4)
        align_loss += self.ts*(align_loss_user + align_loss_item) / 2
        uniform_loss += self.ts*(unif_loss_user + unif_loss_item) / 2

        ssl_loss += self.beta * (align_loss + uniform_loss)

        return user_emb[user_idx], item_emb[item_idx], item_emb[neg_item_idx], ssl_loss,car_loss

        # return user_emb[user_idx], item_emb[item_idx], item_emb[neg_item_idx], \
        #     0, \
        #     0, \
        #     0, ssl_loss

    def bpr_loss(self, users, pos, neg):
        users_emb, pos_emb, neg_emb, cl_loss,car_loss = self.forward(users, pos, neg)
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        loss += self.cl_rate * cl_loss
        loss += car_loss
        return loss, reg_loss

    def predict(self, user_idx, item_idx):
        ua_embedding, ia_embedding, ua_embedding1, ia_embedding1,ua_embedding2, ia_embedding2,int_embeddings = self.LightGCN(self.feature_dict)
        # metapath-based aggregation, h2
        h2 = {}
        for key in self.meta_path_patterns.keys():
            for i in range(self.han_layers):
                if i == 0:
                    h2[key] = self.hans[key](self.g, self.feature_dict[key])
                else:
                    h2[key] = self.hans[key](self.g, h2[key])
        user_emb = 0.5 * ua_embedding + 0.5 * h2[self.user_key]
        item_emb = 0.5 * ia_embedding + 0.5 * h2[self.item_key]
        user_emb = user_emb[user_idx]
        item_emb = item_emb[item_idx]
        return user_emb, item_emb

    def getUsersRating(self, user_idx):
        # x = [2742, 2741, 2743, 2700, 1192, 1976, 2736, 2740, 1201, 2744, 2745, 2739, 2731, 2738, 2161, 1262, 2737, 2722, 2734, 2735, 2733, 2721, 2706, 2729, 849, 2712, 1850, 2732, 2707, 2704, 1853, 2117, 2724, 2551, 2730, 1564, 2726, 2728, 2695, 2727, 2664, 2599, 2718, 2442, 2313, 2716, 2717, 2725, 2554, 2711, 2008, 992, 2698, 2182, 2345, 2713, 2714, 2723, 2719, 2720, 2479, 1430, 1150, 2414, 2642, 2088, 2709, 2648, 1500, 2692, 1436, 2708, 1506, 2701, 2686, 2673, 2705, 2703, 2687, 2267, 2715, 2048, 2689, 2694, 1677, 2683, 2377, 2697, 2702, 2578, 2336, 2489, 2639, 2587, 2688, 2545, 1649, 2653, 2699, 2512, 1018, 2640, 2690, 2107, 2710, 2691, 2669, 2693, 2620, 1719, 2696, 2408, 955, 2649, 1562, 2556, 2362, 1985, 2591, 2680, 1683, 2630, 2681, 2657, 2580, 2679, 2656, 2662, 2607, 2666, 2563, 2651, 2081, 2349, 2685, 2016, 2530, 2558, 2590, 2561, 2627, 2598, 2568, 2663, 2670, 2629, 2575, 2634, 2233, 2659, 2674, 2661, 2641, 2007, 2684, 2402, 1639, 2463, 2682, 2611, 1909, 2660, 1561, 2515, 2595, 2610, 2672, 2645, 2608, 2559, 2652, 2644, 2625, 2617, 2021, 2542, 2655, 2577, 2084, 2643, 1970, 2204, 1676, 2677, 2675, 1442, 2667, 2537, 2628, 2564, 2678, 2613, 1739, 2668, 2658, 2571, 2472, 2676, 2291, 2671, 2665, 2363, 2650, 2635, 1765, 2633, 1779, 2654, 2626, 2615, 2536, 2612, 2525, 2583, 2404, 2400, 2355, 2462, 1937, 2597, 2394, 2570, 2114, 1913, 2356, 2535, 2227, 2621, 2500, 2623, 1299, 2549, 2526, 2646, 1941, 2543, 2596, 2637, 2619, 2638, 2647, 2636, 1174, 2322, 1959, 2170, 1998, 1761, 2293, 2309, 2506, 2268, 2533, 2609, 2490, 2453, 2518, 2366, 2631, 2465, 2602, 2552, 2112, 2352, 2508, 1778, 2614, 1942, 2565, 2624, 1139, 2517, 2606, 2632, 2532, 2138, 2585, 2576, 2618, 2569, 623, 2592, 1245, 2546, 2418, 2622, 2398, 2566, 2616, 2399, 2303, 2338, 1843, 2498, 2514, 2604, 2544, 2448, 2494, 2521, 2478, 930, 2510, 1431, 2409, 2340, 1701, 2555, 2531, 2053, 2522, 2593, 1999, 2105, 2579, 2605, 2449, 2254, 2573, 1588, 2594, 2428, 2452, 2547, 2288, 2553, 2589, 2603, 789, 2523, 1284, 2513, 2135, 2401, 2422, 2582, 2475, 2455, 2492, 843, 1980, 2601, 2541, 2502, 2534, 2213, 2371, 2421, 1965, 2025, 2341, 2295, 1468, 1709, 2560, 2183, 1784, 2314, 2483, 2332, 2420, 2069, 2600, 2180, 2493, 1911, 2584, 2586, 2567, 2588, 2562, 2486, 2347, 2003, 2574, 1990, 2389, 2528, 1275, 1364, 2343, 986, 2524, 2124, 2225, 1242, 2440, 2519, 2415, 2230, 2459, 2250, 2456, 2507, 2433, 2488, 2447, 1809, 2368, 2328, 2405, 2464, 2264, 2503, 2477, 1944, 2384, 2481, 1225, 2520, 2429, 2550, 2473, 1629, 2581, 2424, 2470, 1573, 2548, 1859, 2540, 2538, 2557, 2572, 2294, 1489, 2485, 2484, 2058, 2393, 2443, 2375, 1993, 1953, 2504, 2509, 2511, 2416, 2219, 2299, 2330, 2191, 2457, 2487, 2496, 2495, 2132, 2317, 2407, 1664, 2411, 2430, 2469, 2235, 2306, 1805, 1714, 1700, 2278, 2396, 2461, 2539, 2505, 2256, 2527, 2412, 2207, 1600, 760, 2480, 2381, 2427, 2529, 1735, 2392, 2410, 2186, 1710, 1545, 2413, 944, 2397, 1791, 1974, 2296, 2499, 1625, 2323, 2361, 2467, 2279, 2497, 1637, 2282, 2284, 1617, 2026, 2491, 1493, 1247, 1813, 2471, 2458, 2441, 1394, 1910, 2450, 2468, 2175, 2210, 2325, 2376, 2426, 2027, 2445, 1380, 2482, 2344, 2516, 1323, 2247, 1295, 2476, 432, 2331, 1713, 1628, 2231, 2029, 1962, 2333, 2111, 2365, 2451, 2252, 2241, 2312, 2310, 2370, 2301, 2438, 2185, 2228, 1772, 2382, 2251, 2197, 2168, 2041]
        # item_idx = torch.Tensor(x).long().to(self.device)
        item_idx = torch.Tensor(np.arange(self.inum)).long().to(self.device)
        # users_emb, all_items, _ = self.forward(user_idx, item_idx, None)
        users_emb, all_items = self.predict(user_idx, item_idx)
        rating = torch.matmul(users_emb, all_items.t())
        return rating
