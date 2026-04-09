# rank_videos.py
#https://gemini.google.com/app/f56621858c03837d
import json
import os
import argparse
from typing import List, Dict, Any
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cosine

def get_all_actions(annotation_data: List[Dict]) -> Dict[str, Dict]:
    """遍历所有takes，提取所有原子动作并按规范格式化ID。"""
    all_actions_map = {}
    for take_data in annotation_data:
        take_uid = take_data.get("take_uid")
        if not take_uid: continue
        
        actions = sorted(take_data.get("atomic_descriptions", []), key=lambda x: x["timestamp"])
        for i, action in enumerate(actions):
            action_id = f"{take_uid}_A{i:02d}" # Create a globally unique ID
            all_actions_map[action_id] = {
                "id": action_id,
                "take_uid": take_uid,
                "text": action.get("text", ""),
                "agent": action.get("subject", "UnknownAgent")
            }
    return all_actions_map

def calculate_outlier_scores(
    all_actions_map: Dict[str, Dict], 
    model: SentenceTransformer, 
    method: str = 'centroid', 
    k: int = 5
) -> Dict[str, float]:
    """
    对所有动作进行全局分析，计算每个动作的离群分数。
    """
    print(f"\n--- Calculating Outlier Scores using '{method}' method ---")
    
    action_ids = list(all_actions_map.keys())
    action_texts = [all_actions_map[aid]['text'] for aid in action_ids]
    
    print(f"Encoding {len(action_texts)} unique actions globally...")
    embeddings = model.encode(action_texts, show_progress_bar=True, convert_to_numpy=True)
    
    outlier_scores = {}
    if method == 'centroid':
        # 计算所有向量的质心 (平均向量)
        centroid = np.mean(embeddings, axis=0)
        print("Calculating cosine distance to centroid for each action...")
        for i, emb in enumerate(embeddings):
            # 使用余弦距离 (1 - 余弦相似度) 作为离群分数
            score = cosine(emb, centroid)
            outlier_scores[action_ids[i]] = score
            
    elif method == 'knn':
        print(f"Building K-Nearest Neighbors index (k={k})...")
        # n_neighbors需要k+1，因为每个点自己是自己的最近邻
        neighbors = NearestNeighbors(n_neighbors=k + 1, metric='cosine', algorithm='brute')
        neighbors.fit(embeddings)
        
        print("Calculating average distance to k-nearest neighbors...")
        # 找出每个点到其k个最近邻的距离
        distances, _ = neighbors.kneighbors(embeddings)
        
        # 计算平均距离 (不包括到自己的距离，所以从第1个开始)
        # 距离越大，越离群
        avg_distances = np.mean(distances[:, 1:], axis=1)
        
        for i, avg_dist in enumerate(avg_distances):
            outlier_scores[action_ids[i]] = avg_dist
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
        
    return outlier_scores

def calculate_metrics(
    graph: nx.DiGraph,
    action_to_agent_map: Dict[str, str],
    action_outlier_scores: Dict[str, float]
) -> Dict[str, Any]:
    """为一个视频的因果图计算所有价值指标。"""
    metrics = {
        "node_count": graph.number_of_nodes(), "edge_count": graph.number_of_edges(),
        "causal_depth": 0, "causal_density": 0.0, "interaction_ratio": 0.0,
        "avg_ccn": 0.0, "avg_cnda_score": 0.0
    }
    if not graph.nodes(): return metrics

    if nx.is_directed_acyclic_graph(graph): metrics["causal_depth"] = nx.dag_longest_path_length(graph)
    if graph.number_of_nodes() > 0: metrics["causal_density"] = graph.number_of_edges() / graph.number_of_nodes()

    if graph.number_of_edges() > 0:
        inter_agent_edges = sum(1 for u, v in graph.edges() if action_to_agent_map.get(u) != action_to_agent_map.get(v))
        metrics["interaction_ratio"] = inter_agent_edges / graph.number_of_edges()

    try:
        ccn_values = nx.betweenness_centrality(graph, normalized=True)
        if ccn_values: metrics["avg_ccn"] = np.mean(list(ccn_values.values()))
    except Exception: pass
    
    # NEW: CNDA score is the average outlier score of the nodes in this graph
    node_scores = [action_outlier_scores.get(node_id, 0.0) for node_id in graph.nodes()]
    if node_scores:
        metrics["avg_cnda_score"] = np.mean(node_scores)
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Filter and rank videos using unsupervised outlier detection for CNDA.")
    parser.add_argument('--graph_file', type=str, required=True, help='Input JSON file from the generate_graph phase.')
    parser.add_argument('--annotation_file', type=str, required=True, help='The original annotation file.')
    parser.add_argument('--output_file', type=str, default='ranked_videos_unsupervised.json', help='Output file.')
    
    parser.add_argument('--cnda_method', type=str, default='centroid', choices=['centroid', 'knn'], help='Unsupervised method for CNDA.')
    parser.add_argument('--knn_k', type=int, default=5, help='Number of neighbors for KNN method.')
    
    parser.add_argument('--min_depth', type=int, default=3, help='Minimum causal depth.')
    parser.add_argument('--min_density', type=float, default=0.8, help='Minimum causal density.')
    parser.add_argument('--min_avg_ccn', type=float, default=0.05, help='Minimum average betweenness centrality.')
    parser.add_argument('--min_avg_cnda', type=float, default=0.3, help='Minimum average CNDA outlier score.')
    args = parser.parse_args()

    # 1. 初始化模型
    print("Loading SentenceTransformer model...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    # 2. 加载数据
    with open(args.graph_file, 'r', encoding='utf-8') as f: graphs_data = json.load(f)
    with open(args.annotation_file, 'r', encoding='utf-8') as f: all_takes_data = json.load(f)
    if not isinstance(all_takes_data, list): all_takes_data = [all_takes_data]
    
    # 3. 全局分析：为数据集中所有动作计算离群分数
    all_actions_map = get_all_actions(all_takes_data)
    action_outlier_scores = calculate_outlier_scores(all_actions_map, sbert_model, args.cnda_method, args.knn_k)

    # 4. 筛选与指标计算
    suitable_videos, rejected_videos = [], []
    print("\n--- Running Filtering and Metric Calculation for each video ---")
    for take_uid, edges in graphs_data.items():
        print(f"Processing {take_uid}...")
        
        # 构建图和上下文
        g = nx.DiGraph()
        node_ids_in_graph = set()
        for edge in edges:
            g.add_edge(edge['from'], edge['to'])
            node_ids_in_graph.add(edge['from'])
            node_ids_in_graph.add(edge['to'])
        
        # 为当前 take 创建局部的 action -> agent 映射
        take_actions = [action for action in all_actions_map.values() if action['take_uid'] == take_uid]
        action_to_agent_map = {action['id']: action['agent'] for action in take_actions}
        
        metrics = calculate_metrics(g, [], action_to_agent_map, action_outlier_scores)
        
        is_suitable = (
            metrics["causal_depth"] >= args.min_depth and
            metrics["causal_density"] >= args.min_density and
            metrics["avg_ccn"] >= args.min_avg_ccn and
            metrics["avg_cnda_score"] >= args.min_avg_cnda
        )
        
        video_info = {"take_uid": take_uid, "metrics": metrics}
        if is_suitable:
            suitable_videos.append(video_info); print(f"  -> ✅ SUITABLE. Avg CNDA Score: {metrics['avg_cnda_score']:.3f}")
        else:
            rejected_videos.append(video_info); print(f"  -> ❌ REJECTED. Avg CNDA Score: {metrics['avg_cnda_score']:.3f}")

    # 5. 对筛选出的视频按“团队协作复杂度”进行难度排序
    print(f"\n--- Ranking {len(suitable_videos)} Suitable Videos by Teamwork Complexity ---")
    suitable_videos.sort(key=lambda x: x['metrics']['interaction_ratio'], reverse=True)
    for i, video in enumerate(suitable_videos):
        video['difficulty_rank'] = i + 1; video['difficulty_score (interaction_ratio)'] = video['metrics']['interaction_ratio']

    # 6. 保存结果
    with open(args.output_file, 'w', encoding='utf-8') as f: json.dump(suitable_videos, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Ranking complete! Results saved to {args.output_file}")
    print("Top 5 most difficult (highest teamwork) suitable videos:")
    for video in suitable_videos[:5]: print(f"  Rank {video['difficulty_rank']}: {video['take_uid']} (Teamwork Score: {video['difficulty_score (interaction_ratio)']:.4f})")

if __name__ == '__main__':
    main()


# rank_videos.py

import json
import os
import argparse
from typing import List, Dict, Any
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cosine

def get_all_actions(annotation_data: List[Dict]) -> Dict[str, Dict]:
    """
    遍历所有takes，提取所有原子动作并创建全局唯一的ID和映射。
    e.g., {"0660616d-d2b4-499a-91df-988c0bcaefba_A01": {"id": ..., "text": ...}}
    """
    all_actions_map = {}
    for take_data in annotation_data:
        take_uid = take_data.get("take_uid")
        if not take_uid: continue
        
        actions = sorted(take_data.get("atomic_descriptions", []), key=lambda x: x["timestamp"])
        for i, action in enumerate(actions):
            action_id = f"{take_uid}_A{i:02d}" # 创建全局唯一ID
            all_actions_map[action_id] = {
                "id": action_id,
                "take_uid": take_uid,
                "text": action.get("text", ""),
                "agent": action.get("subject", "UnknownAgent")
            }
    return all_actions_map

def calculate_outlier_scores(
    all_actions_map: Dict[str, Dict], 
    model: SentenceTransformer, 
    method: str = 'centroid', 
    k: int = 5
) -> Dict[str, float]:
    """
    对所有动作进行全局分析，计算每个动作的离群分数。
    """
    print(f"\n--- Calculating Outlier Scores using '{method}' method ---")
    
    action_ids = list(all_actions_map.keys())
    action_texts = [all_actions_map[aid]['text'] for aid in action_ids]
    
    print(f"Encoding {len(action_texts)} unique actions globally...")
    embeddings = model.encode(action_texts, show_progress_bar=True, convert_to_numpy=True)
    
    outlier_scores = {}
    if method == 'centroid':
        centroid = np.mean(embeddings, axis=0)
        print("Calculating cosine distance to centroid for each action...")
        for i, emb in enumerate(embeddings):
            score = cosine(emb, centroid)
            outlier_scores[action_ids[i]] = score
            
    elif method == 'knn':
        print(f"Building K-Nearest Neighbors index (k={k})...")
        neighbors = NearestNeighbors(n_neighbors=k + 1, metric='cosine', algorithm='brute')
        neighbors.fit(embeddings)
        print("Calculating average distance to k-nearest neighbors...")
        distances, _ = neighbors.kneighbors(embeddings)
        avg_distances = np.mean(distances[:, 1:], axis=1)
        for i, avg_dist in enumerate(avg_distances):
            outlier_scores[action_ids[i]] = avg_dist
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
        
    return outlier_scores

def calculate_metrics(
    graph: nx.DiGraph,
    action_to_agent_map: Dict[str, str],
    action_outlier_scores: Dict[str, float],
    take_uid: str
) -> Dict[str, Any]:
    """为一个视频的因果图计算所有价值指标。"""
    metrics = {
        "node_count": graph.number_of_nodes(), "edge_count": graph.number_of_edges(),
        "causal_depth": 0, "causal_density": 0.0, "interaction_ratio": 0.0,
        "avg_ccn": 0.0, "avg_cnda_score": 0.0
    }
    if not graph.nodes(): return metrics

    if nx.is_directed_acyclic_graph(graph): metrics["causal_depth"] = nx.dag_longest_path_length(graph)
    if graph.number_of_nodes() > 0: metrics["causal_density"] = graph.number_of_edges() / graph.number_of_nodes()

    if graph.number_of_edges() > 0:
        inter_agent_edges = sum(1 for u, v in graph.edges() if action_to_agent_map.get(u) != action_to_agent_map.get(v))
        metrics["interaction_ratio"] = inter_agent_edges / graph.number_of_edges()

    try:
        ccn_values = nx.betweenness_centrality(graph, normalized=True)
        if ccn_values: metrics["avg_ccn"] = np.mean(list(ccn_values.values()))
    except Exception: pass
    
    # CNDA score is the average outlier score of the nodes in this graph
    node_scores = [action_outlier_scores.get(f"{take_uid}_{node_id}", 0.0) for node_id in graph.nodes()]
    if node_scores:
        metrics["avg_cnda_score"] = np.mean(node_scores)
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Filter and rank videos using unsupervised outlier detection for CNDA.")
    parser.add_argument('--graph_file', type=str, required=True, help='Input JSON file from the generate_graph phase.')
    parser.add_argument('--annotation_file', type=str, required=True, help='The original annotation file.')
    parser.add_argument('--output_file', type=str, default='ranked_videos_unsupervised.json', help='Output file.')
    
    parser.add_argument('--cnda_method', type=str, default='centroid', choices=['centroid', 'knn'], help='Unsupervised method for CNDA.')
    parser.add_argument('--knn_k', type=int, default=5, help='Number of neighbors for KNN method.')
    
    parser.add_argument('--min_depth', type=int, default=3, help='Minimum causal depth.')
    parser.add_argument('--min_density', type=float, default=0.8, help='Minimum causal density.')
    parser.add_argument('--min_avg_ccn', type=float, default=0.05, help='Minimum average betweenness centrality.')
    parser.add_argument('--min_avg_cnda', type=float, default=0.3, help='Minimum average CNDA outlier score.')
    args = parser.parse_args()

    # 1. 初始化模型
    print("Loading SentenceTransformer model...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    # 2. 加载数据
    with open(args.graph_file, 'r', encoding='utf-8') as f: graphs_data = json.load(f)
    with open(args.annotation_file, 'r', encoding='utf-8') as f: all_takes_data = json.load(f)
    if not isinstance(all_takes_data, list): all_takes_data = [all_takes_data]
    
    # 3. 全局分析：为数据集中所有动作计算离群分数
    all_actions_map = get_all_actions(all_takes_data)
    action_outlier_scores = calculate_outlier_scores(all_actions_map, sbert_model, args.cnda_method, args.knn_k)
    
    take_data_map = {t.get("take_uid"): t for t in all_takes_data}

    # 4. 筛选与指标计算
    suitable_videos, rejected_videos = [], []
    print("\n--- Running Filtering and Metric Calculation for each video ---")
    for take_uid, edges in graphs_data.items():
        print(f"Processing {take_uid}...")
        
        # MODIFIED: 智能跳过空边列表
        if not edges:
            print(f"  -> 🟡 SKIPPED. The edge list for this UID is empty.")
            continue

        take_data = take_data_map.get(take_uid)
        if not take_data: 
            print(f"  -> 🟡 SKIPPED. Could not find original annotation data for this UID.")
            continue

        # 构建图和上下文
        g = nx.DiGraph()
        # MODIFIED: 兼容 final_causal_graph.json 的格式
        for edge in edges:
            from_id = edge.get("from_action", {}).get("id")
            to_id = edge.get("to_action", {}).get("id")
            if from_id and to_id:
                g.add_edge(from_id, to_id)

        actions_in_take = sorted(take_data.get("atomic_descriptions", []), key=lambda x: x["timestamp"])
        actions_in_take = [{"id": f"A{i:02d}", **a} for i, a in enumerate(actions_in_take)]
        action_to_agent_map = {action['id']: action.get("subject", "UnknownAgent") for action in actions_in_take}
        
        metrics = calculate_metrics(g, [], action_to_agent_map, action_outlier_scores, take_uid)
        
        is_suitable = (
            metrics["causal_depth"] >= args.min_depth and
            metrics["causal_density"] >= args.min_density and
            metrics["avg_ccn"] >= args.min_avg_ccn and
            metrics["avg_cnda_score"] >= args.min_avg_cnda
        )
        
        video_info = {"take_uid": take_uid, "metrics": metrics}
        if is_suitable:
            suitable_videos.append(video_info); print(f"  -> ✅ SUITABLE. Avg CNDA Score: {metrics['avg_cnda_score']:.3f}")
        else:
            rejected_videos.append(video_info); print(f"  -> ❌ REJECTED. Avg CNDA Score: {metrics['avg_cnda_score']:.3f}")

    # 5. 对筛选出的视频按“团队协作复杂度”进行难度排序
    print(f"\n--- Ranking {len(suitable_videos)} Suitable Videos by Teamwork Complexity ---")
    suitable_videos.sort(key=lambda x: x['metrics']['interaction_ratio'], reverse=True)
    for i, video in enumerate(suitable_videos):
        video['difficulty_rank'] = i + 1; video['difficulty_score (interaction_ratio)'] = video['metrics']['interaction_ratio']

    # 6. 保存结果
    with open(args.output_file, 'w', encoding='utf-8') as f: json.dump(suitable_videos, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Ranking complete! Results saved to {args.output_file}")
    print("Top 5 most difficult (highest teamwork) suitable videos:")
    for video in suitable_videos[:5]: print(f"  Rank {video['difficulty_rank']}: {video['take_uid']} (Teamwork Score: {video['difficulty_score (interaction_ratio)']:.4f})")

if __name__ == '__main__':
    main()