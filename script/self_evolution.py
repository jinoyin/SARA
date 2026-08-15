import os
import sys
import copy
import math
import pprint
import random
import shutil
from collections import defaultdict
from itertools import islice
from functools import partial

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ultra import tasks, util
from ultra.models import Ultra
from ultra.tasks import build_relation_graph

import numpy as np

separator = ">" * 30
line = "-" * 30


# ══════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════

def save_train_triplets_text(graph, id2entity, id2relation, save_path):
    edge_index = graph.target_edge_index
    edge_type  = graph.target_edge_type
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for (h, t), r in zip(edge_index.t().tolist(), edge_type.tolist()):
            f.write(f"{id2entity[h]}\t{id2relation[r]}\t{id2entity[t]}\n")


def multigraph_collator(batch, train_graphs):
    probs = torch.tensor([g.edge_index.shape[1] for g in train_graphs]).float()
    probs /= probs.sum()
    graph_id  = torch.multinomial(probs, 1, replacement=False).item()
    graph     = train_graphs[graph_id]
    bs        = len(batch)
    edge_mask = torch.randperm(graph.target_edge_index.shape[1])[:bs]
    batch = torch.cat([
        graph.target_edge_index[:, edge_mask],
        graph.target_edge_type[edge_mask].unsqueeze(0)
    ]).t()
    return graph, batch


# ══════════════════════════════════════════════════════════════════
# 加载预训练模型
# ══════════════════════════════════════════════════════════════════

def load_pretrained_model(cfg, pretrain_ckpt, device):
    model = Ultra(
        rel_model_cfg=copy.deepcopy(cfg.model.relation_model),
        entity_model_cfg=copy.deepcopy(cfg.model.entity_model),
    )
    if pretrain_ckpt is not None:
        if not os.path.exists(pretrain_ckpt):
            raise FileNotFoundError(
                f"[load_pretrained_model] checkpoint not found: {pretrain_ckpt}"
            )
        state   = torch.load(pretrain_ckpt, map_location="cpu")
        weights = state.get("model", state)
        model.load_state_dict(weights)
        if util.get_rank() == 0:
            logger.warning(f"[load_pretrained_model] Loaded weights from: {pretrain_ckpt}")
    else:
        if util.get_rank() == 0:
            logger.warning("[load_pretrained_model] No checkpoint -> random initialization")
    return model.to(device)


def load_verify_models(cfg, verify_ckpts, device):
    if not verify_ckpts:
        return []
    models = []
    for idx, ckpt in enumerate(verify_ckpts):
        if util.get_rank() == 0:
            logger.warning(
                f"[load_verify_models] Loading verify model "
                f"{idx + 1}/{len(verify_ckpts)}: {ckpt}"
            )
        m = load_pretrained_model(cfg, ckpt, device)
        m.eval()
        models.append(m)
    if util.get_rank() == 0:
        logger.warning(f"[load_verify_models] Loaded {len(models)} verify model(s).")
    return models


# ══════════════════════════════════════════════════════════════════
# 图增强：单关系对称 + 关系共现补全
#
#   对称补全   (h, r, t) => (t, r, h)
#   共现补全   (h, r1, t) 与 (h, r2, t) 高频共现 => 补全缺少的关系边
#
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def augment_and_filter_edges(
    cfg, model, train_data,
    sym_ratio_threshold=0.6,
    sym_score_threshold=0.4,
    cooccur_ratio_threshold=0.6,
    min_cooccur_support=5,
    cooccur_score_threshold=0.4,
):
    """
    同时执行单关系对称和关系共现两类图结构增强，并用模型分数过滤候选边。

    共现补全核心逻辑：
      base_cnt[r]           = |{(h,t) : (h,r,t) in E}|
      cooccur_cnt[r1][r2]   = |pair_by_rel[r1] & pair_by_rel[r2]|
      筛选条件（双向均满足）：
        cooccur_cnt[r1][r2] >= min_cooccur_support
        cooccur_cnt[r1][r2] / base_cnt[r1] >= cooccur_ratio_threshold
        cooccur_cnt[r2][r1] / base_cnt[r2] >= cooccur_ratio_threshold
      生成候选：
        (h,r1,t) in E 且 (h,r2,t) not in E => 候选 (h, r2, t)，反之亦然
    """
    rank = util.get_rank()
    model.eval()

    if rank == 0:
        logger.warning(
            "  [augment] methods=single-relation symmetry + relation co-occurrence"
        )

    augmented_train_data = []

    for graph_idx, train_graph in enumerate(train_data):
        if rank == 0:
            logger.warning(f"Processing graph {graph_idx + 1}/{len(train_data)}")

        edge_index    = train_graph.target_edge_index
        edge_type     = train_graph.target_edge_type
        num_edges     = edge_index.shape[1]
        num_relations = train_graph.edge_type.max().item() // 2

        existing_edges = set(zip(
            edge_index[0].tolist(),
            edge_index[1].tolist(),
            edge_type.tolist(),
        ))
        original_edges = copy.copy(existing_edges)
        relations      = torch.unique(edge_type).tolist()

        # ── 两种增强方法共用的索引结构 ───────────────────────────
        # pair_to_rels[(h, t)] = {r, ...}   — 该 (h,t) 对上存在的所有关系
        # pair_by_rel[r]       = {(h,t),...} — 关系 r 覆盖的所有 (h,t) 对
        pair_to_rels = defaultdict(set)
        pair_by_rel  = defaultdict(set)
        for h, t, r in original_edges:
            pair_to_rels[(h, t)].add(r)
            pair_by_rel[r].add((h, t))

        sym_selected_edges = torch.empty(
            (0, 3), device=edge_index.device, dtype=torch.long
        )
        cooccur_selected_edges = torch.empty(
            (0, 3), device=edge_index.device, dtype=torch.long
        )

        # ══════════════════════════════════════════════════════════
        # 单关系对称补全
        #   统计：sym_ratio = |(h,r,t)&(t,r,h)| / |(h,r,t)|
        #   筛选：r 满足 sym_ratio >= sym_ratio_threshold
        #   候选：(h,r,t) in E 且 (t,r,h) not in E => 候选 (t,r,h)
        # ══════════════════════════════════════════════════════════
        if rank == 0:
            logger.warning("[单关系对称] 识别对称关系并生成补全候选...")

        total_cnt, sym_cnt = {}, {}
        for h, t, r in original_edges:
            total_cnt[r] = total_cnt.get(r, 0) + 1
            if (t, h, r) in original_edges:
                sym_cnt[r] = sym_cnt.get(r, 0) + 1

        symmetric_relations = {
            r for r, count in sym_cnt.items()
            if count / max(total_cnt.get(r, 1), 1) >= sym_ratio_threshold
        }
        if rank == 0:
            logger.warning(
                f"  发现对称关系 {len(symmetric_relations)}/{len(relations)} 个: "
                f"{symmetric_relations}"
            )

        sym_candidate_set = set()
        for h, t, r in original_edges:
            if r in symmetric_relations and (t, h, r) not in original_edges:
                sym_candidate_set.add((t, h, r))

        if rank == 0:
            logger.warning(f"  单关系对称候选数量: {len(sym_candidate_set)}")

        if sym_candidate_set:
            sym_edges_t = torch.tensor(
                list(sym_candidate_set), device=edge_index.device, dtype=torch.long
            )
            passed = []
            for i in range(0, sym_edges_t.size(0), cfg.train.batch_size):
                query_batch = sym_edges_t[i : i + cfg.train.batch_size]
                scores = torch.sigmoid(
                    model(train_graph, query_batch.unsqueeze(1))[:, 0]
                )
                mask = scores >= sym_score_threshold
                if mask.any():
                    passed.append(query_batch[mask])
            if passed:
                sym_selected_edges = torch.cat(passed, dim=0)
                if rank == 0:
                    logger.warning(
                        f"  单关系对称通过打分: "
                        f"{sym_selected_edges.size(0)} / {sym_edges_t.size(0)}"
                    )
            elif rank == 0:
                logger.warning("  单关系对称：无候选通过打分阈值")

        for row in sym_selected_edges.tolist():
            existing_edges.add(tuple(row))

        # ══════════════════════════════════════════════════════════
        # 关系共现补全
        #
        # 直觉：若 r1 和 r2 对相同的 (h,t) 对高频共现，
        #       则二者语义近似，可用对方覆盖自己遗漏的边。
        #
        # 统计：
        #   base_cnt[r]         = |pair_by_rel[r]|
        #   cooccur_cnt[r1][r2] = |pair_by_rel[r1] & pair_by_rel[r2]|
        #
        # 双向筛选条件（防止大关系单向吞并小关系）：
        #   cooccur_cnt[r1][r2] >= min_cooccur_support
        #   cooccur_cnt[r1][r2] / base_cnt[r1] >= cooccur_ratio_threshold
        #   cooccur_cnt[r2][r1] / base_cnt[r2] >= cooccur_ratio_threshold
        #
        # 候选生成（双向互补）：
        #   (h,r1,t) in E 且 (h,r2,t) not in E => 候选 (h, r2, t)
        #   (h,r2,t) in E 且 (h,r1,t) not in E => 候选 (h, r1, t)
        # ══════════════════════════════════════════════════════════
        if rank == 0:
            logger.warning("[关系共现] 挖掘共现关系对并生成补全候选...")

        # 各关系覆盖的 (h,t) 对数
        base_cnt = {r: len(pair_by_rel[r]) for r in relations}

        # 遍历所有 (h,t)，对其关系集合做两两共现统计
        # 复杂度 O(|E| * avg_degree_per_pair^2)，通常 avg_degree 很小
        cooccur_cnt = defaultdict(lambda: defaultdict(int))
        for rels in pair_to_rels.values():
            rels_list = list(rels)
            for i in range(len(rels_list)):
                for j in range(i + 1, len(rels_list)):
                    r1, r2 = rels_list[i], rels_list[j]
                    cooccur_cnt[r1][r2] += 1
                    cooccur_cnt[r2][r1] += 1

        # 筛选满足双向阈值的共现关系对
        cooccur_relation_pairs = []
        cooccur_stats = []
        seen_pairs = set()

        for r1 in relations:
            for r2, count in cooccur_cnt[r1].items():
                if r1 == r2 or (r2, r1) in seen_pairs:
                    continue
                seen_pairs.add((r1, r2))

                base_r1 = max(base_cnt.get(r1, 1), 1)
                base_r2 = max(base_cnt.get(r2, 1), 1)
                ratio_r1 = count / base_r1
                ratio_r2 = cooccur_cnt[r2].get(r1, 0) / base_r2

                if (
                    count >= min_cooccur_support
                    and ratio_r1 >= cooccur_ratio_threshold
                    and ratio_r2 >= cooccur_ratio_threshold
                ):
                    cooccur_relation_pairs.append((r1, r2))
                    cooccur_stats.append(
                        (r1, r2, count, base_r1, base_r2, ratio_r1, ratio_r2)
                    )

        if rank == 0:
            logger.warning(
                f"  发现共现关系对 {len(cooccur_relation_pairs)} 对 "
                f"(ratio>={cooccur_ratio_threshold:.2f}, "
                f"support>={min_cooccur_support})"
            )
            for r1, r2, count, base_r1, base_r2, ratio_r1, ratio_r2 in sorted(
                cooccur_stats,
                key=lambda item: (item[5] + item[6]) / 2,
                reverse=True,
            ):
                logger.warning(
                    f"    r{r1} ~ r{r2} "
                    f"(共现={count}, base_r1={base_r1}, base_r2={base_r2}, "
                    f"ratio_r1={ratio_r1:.1%}, ratio_r2={ratio_r2:.1%})"
                )

        # 双向互补生成候选
        cooccur_candidate_set = set()
        for r1, r2 in cooccur_relation_pairs:
            pairs_r1 = pair_by_rel[r1]
            pairs_r2 = pair_by_rel[r2]

            # r1 有、r2 没有 => 候选 (h, r2, t)
            for h, t in pairs_r1 - pairs_r2:
                candidate = (h, t, r2)
                if candidate not in existing_edges:
                    cooccur_candidate_set.add(candidate)

            # r2 有、r1 没有 => 候选 (h, r1, t)
            for h, t in pairs_r2 - pairs_r1:
                candidate = (h, t, r1)
                if candidate not in existing_edges:
                    cooccur_candidate_set.add(candidate)

        if rank == 0:
            logger.warning(f"  共现补全候选数量: {len(cooccur_candidate_set)}")

        if cooccur_candidate_set:
            cooccur_edges_t = torch.tensor(
                list(cooccur_candidate_set),
                device=edge_index.device,
                dtype=torch.long,
            )
            passed = []
            for i in range(0, cooccur_edges_t.size(0), cfg.train.batch_size):
                query_batch = cooccur_edges_t[i : i + cfg.train.batch_size]
                scores = torch.sigmoid(
                    model(train_graph, query_batch.unsqueeze(1))[:, 0]
                )
                mask = scores >= cooccur_score_threshold
                if mask.any():
                    passed.append(query_batch[mask])
            if passed:
                cooccur_selected_edges = torch.cat(passed, dim=0)
                if rank == 0:
                    logger.warning(
                        f"  共现补全通过打分: "
                        f"{cooccur_selected_edges.size(0)} / {cooccur_edges_t.size(0)}"
                    )
            elif rank == 0:
                logger.warning("  共现补全：无候选通过打分阈值")

        for row in cooccur_selected_edges.tolist():
            existing_edges.add(tuple(row))

        # ══════════════════════════════════════════════════════════
        # 合并两类补全结果
        # ══════════════════════════════════════════════════════════
        selected_edges = torch.cat(
            [
                sym_selected_edges,
                cooccur_selected_edges,
            ],
            dim=0,
        )

        if rank == 0:
            logger.warning(
                f"  Added {selected_edges.size(0)} edges total | "
                f"单关系对称={sym_selected_edges.size(0)} | "
                f"共现补全={cooccur_selected_edges.size(0)}"
            )

        # ══════════════════════════════════════════════════════════
        # Step F：拼接图
        # ══════════════════════════════════════════════════════════
        final_edge_index = torch.cat(
            [edge_index, selected_edges[:, :2].t()], dim=1
        )
        final_edge_type = torch.cat(
            [edge_type, selected_edges[:, 2]], dim=0
        )

        augmented_graph                   = copy.deepcopy(train_graph)
        augmented_graph.target_edge_index = final_edge_index
        augmented_graph.target_edge_type  = final_edge_type

        reverse_edge_index = final_edge_index.flip(0)
        reverse_edge_type  = final_edge_type + num_relations
        augmented_graph.edge_index = torch.cat(
            [final_edge_index, reverse_edge_index], dim=1
        )
        augmented_graph.edge_type = torch.cat(
            [final_edge_type, reverse_edge_type], dim=0
        )

        device_graph    = augmented_graph.edge_index.device
        augmented_graph = augmented_graph.to("cpu")
        augmented_graph = build_relation_graph(augmented_graph)
        augmented_graph = augmented_graph.to(device_graph)

        if rank == 0:
            logger.warning(
                f"Final: {num_edges} -> {final_edge_index.shape[1]} edges "
                f"(+{selected_edges.size(0)})"
            )
        augmented_train_data.append(augmented_graph)

    return augmented_train_data


# ══════════════════════════════════════════════════════════════════
# 自演化主循环
#
# 每轮完成训练和评估后，同时执行单关系对称与关系共现增强，
# 将增强后的 train.txt 用于下一轮训练。
# ══════════════════════════════════════════════════════════════════

def self_evolution_train(
    cfg, args,
    pretrain_ckpt,
    max_iterations=10,
    patience=3,
    # ── 单关系对称 ───────────────────────────────────────────────
    sym_ratio_threshold=0.6,
    sym_score_threshold=0.4,
    # ── 共现补全 ─────────────────────────────────────────────────
    cooccur_ratio_threshold=0.6,
    min_cooccur_support=5,
    cooccur_score_threshold=0.4,
):
    cfg    = copy.deepcopy(cfg)
    rank   = util.get_rank()
    device = util.get_device(cfg)

    best_valid_score     = float("-inf")
    best_model_state     = None
    no_improvement_count = 0
    best_filtered_data   = None

    if len(cfg.dataset.graphs) != 1:
        raise ValueError("self_evolution_train currently requires exactly one graph")

    dataset_root = os.path.abspath(os.path.expanduser(cfg.dataset.root))
    graph_name = cfg.dataset.graphs[0]
    train_txt_path = os.path.join(dataset_root, graph_name, "train.txt")
    best_train_txt_path = os.path.join(dataset_root, graph_name, "train_best.txt")

    if pretrain_ckpt is not None and not os.path.exists(pretrain_ckpt):
        raise FileNotFoundError(f"pretrain_ckpt not found: {pretrain_ckpt}")

    if rank == 0:
        logger.warning(f"[self_evo] pretrain_ckpt: {pretrain_ckpt}")
        logger.warning(
            "[self_evo] 每次增强同时执行: 单关系对称 + 关系共现补全"
        )

    # ── 初始化 best_train.txt ─────────────────────────────────────
    if rank == 0 and not os.path.exists(best_train_txt_path):
        shutil.copy2(train_txt_path, best_train_txt_path)
        logger.warning("[init] Initialized best_train.txt <- train.txt")
    util.synchronize()

    for iteration in range(max_iterations):
        # ── 随机种子 ──────────────────────────────────────────────
        seed_offset = args.seed + util.get_rank() + iteration * 1024
        random.seed(seed_offset)
        np.random.seed(seed_offset)
        torch.manual_seed(seed_offset)
        torch.cuda.manual_seed(seed_offset)
        torch.cuda.manual_seed_all(seed_offset)
        torch.backends.cudnn.benchmark     = False
        torch.backends.cudnn.deterministic = True

        if rank == 0:
            logger.warning("=" * 60)
            logger.warning(
                f"Self-Evolution Iteration {iteration + 1}/{max_iterations}  "
                "[本轮增强写盘 -> 下一轮训练: 对称 + 共现]"
            )
            logger.warning("=" * 60)

        # ── 删缓存 ───────────────────────────────────────────────
        if rank == 0:
            for cache_dir in [
                os.path.join(dataset_root, cfg.dataset.graphs[0], "processed"),
                os.path.join(dataset_root, cfg.dataset.graphs[0], "raw"),
                os.path.join(dataset_root, "joint", "1g", "processed"),
            ]:
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir)
                    logger.warning(f"Deleted cache: {cache_dir}")
        util.synchronize()

        # ── 加载数据 ──────────────────────────────────────────────
        dataset = util.build_dataset(cfg)
        train_data, valid_data, test_data = (
            dataset._data[0], dataset._data[1], dataset._data[2]
        )
        train_data = [td.to(device) for td in train_data]
        valid_data = [vd.to(device) for vd in valid_data]
        test_data  = [tst.to(device) for tst in test_data]

        if rank == 0:
            logger.warning(
                f"Loaded train edges: {train_data[0].target_edge_index.shape[1]}"
            )

        # ── 从固定 pretrain_ckpt 加载模型 ─────────────────────────
        model = load_pretrained_model(cfg, pretrain_ckpt, device)

        # ── filtered_data ─────────────────────────────────────────
        filtered_data = [
            Data(
                edge_index=torch.cat([
                    trg.target_edge_index,
                    valg.target_edge_index,
                    testg.target_edge_index,
                ], dim=1),
                edge_type=torch.cat([
                    trg.target_edge_type,
                    valg.target_edge_type,
                    testg.target_edge_type,
                ]),
                num_nodes=trg.num_nodes,
            ).to(device)
            for trg, valg, testg in zip(train_data, valid_data, test_data)
        ]

        # ── 训练 ──────────────────────────────────────────────────
        train_and_validate(
            cfg, model, train_data, valid_data,
            filtered_data=filtered_data,
            batch_per_epoch=cfg.train.batch_per_epoch,
            save_prefix=f"iter_{iteration}_",
        )

        # ── 评估 ──────────────────────────────────────────────────
        valid_score = test(cfg, model, valid_data, filtered_data=filtered_data)
        test_score  = test(cfg, model, test_data,  filtered_data=filtered_data)

        if rank == 0:
            logger.warning(
                f"Iteration {iteration + 1} | "
                f"valid={valid_score:.4f} | test={test_score:.4f}"
            )

        # ── 判断是否提升 ──────────────────────────────────────────
        if valid_score > best_valid_score:
            best_valid_score     = valid_score
            best_model_state     = copy.deepcopy(model.state_dict())
            best_filtered_data   = filtered_data
            no_improvement_count = 0
            if rank == 0:
                logger.warning(f"New best valid={best_valid_score:.4f}")
                torch.save(
                    {
                        "model":       best_model_state,
                        "iteration":   iteration,
                        "valid_score": best_valid_score,
                    },
                    "best_self_evolution_model.pth",
                )
                shutil.copy2(train_txt_path, best_train_txt_path)
                logger.warning("Updated best_train.txt <- train.txt")
        else:
            no_improvement_count += 1
            if rank == 0:
                logger.warning(
                    f"No improvement ({no_improvement_count}/{patience})"
                )
                shutil.copy2(best_train_txt_path, train_txt_path)
                logger.warning("Rolled back train.txt <- best_train.txt")

        # ── 早停 ──────────────────────────────────────────────────
        if no_improvement_count >= patience:
            if rank == 0:
                logger.warning("Early stopping triggered")
            util.synchronize()
            break

        # ── 增强阶段：同时执行两种规则，写盘给下一轮 ─────────────
        if iteration < max_iterations - 1 and best_model_state is not None:
            if rank == 0:
                logger.warning(
                    f"[augment] iteration={iteration} | methods=对称 + 共现"
                )

            # 删缓存，保证读到最新 train.txt
            if rank == 0:
                for cache_dir in [
                    os.path.join(dataset_root, cfg.dataset.graphs[0], "processed"),
                    os.path.join(dataset_root, cfg.dataset.graphs[0], "raw"),
                    os.path.join(dataset_root, "joint", "1g", "processed"),
                ]:
                    if os.path.exists(cache_dir):
                        shutil.rmtree(cache_dir)
            util.synchronize()

            dataset_for_aug    = util.build_dataset(cfg)
            train_data_for_aug = [
                dataset_for_aug._data[0][i].to(device)
                for i in range(len(dataset_for_aug._data[0]))
            ]

            best_model = load_pretrained_model(cfg, pretrain_ckpt, device)
            best_model.load_state_dict(best_model_state)

            augmented_data = augment_and_filter_edges(
                cfg, best_model, train_data_for_aug,
                # 单关系对称
                sym_ratio_threshold=sym_ratio_threshold,
                sym_score_threshold=sym_score_threshold,
                # 关系共现
                cooccur_ratio_threshold=cooccur_ratio_threshold,
                min_cooccur_support=min_cooccur_support,
                cooccur_score_threshold=cooccur_score_threshold,
            )

            if rank == 0:
                for idx, g in enumerate(augmented_data):
                    src = (
                        dataset_for_aug.graphs[idx]
                        if hasattr(dataset_for_aug, "graphs")
                        else dataset_for_aug
                    )
                    id2entity   = {v: k for k, v in src.entity2id.items()}
                    id2relation = {v: k for k, v in src.relation2id.items()}
                    save_train_triplets_text(
                        g.to("cpu"),
                        id2entity=id2entity,
                        id2relation=id2relation,
                        save_path=train_txt_path,
                    )
                    logger.warning(
                        f"Written augmented train.txt: "
                        f"{g.target_edge_index.shape[1]} edges "
                        "[对称 + 共现]"
                    )

        util.synchronize()

    # ── 返回最佳模型 ──────────────────────────────────────────────
    if best_model_state is None:
        if rank == 0:
            logger.warning("[Warning] No improvement. Returning pretrained model.")
        final_model = load_pretrained_model(cfg, pretrain_ckpt, device)
    else:
        final_model = load_pretrained_model(cfg, pretrain_ckpt, device)
        final_model.load_state_dict(best_model_state)

    return final_model, best_valid_score, best_filtered_data


# ══════════════════════════════════════════════════════════════════
# 训练 & 测试
# ══════════════════════════════════════════════════════════════════

def train_and_validate(
    cfg, model, train_data, valid_data,
    filtered_data=None, batch_per_epoch=None, save_prefix=""
):
    device     = util.get_device(cfg)
    world_size = util.get_world_size()
    rank       = util.get_rank()

    if cfg.train.num_epoch == 0:
        return

    train_triplets = torch.cat([
        torch.cat([
            g.target_edge_index,
            g.target_edge_type.unsqueeze(0),
        ]).t()
        for g in train_data
    ])
    sampler      = torch_data.DistributedSampler(train_triplets, world_size, rank)
    train_loader = torch_data.DataLoader(
        train_triplets, cfg.train.batch_size, sampler=sampler,
        collate_fn=partial(multigraph_collator, train_graphs=train_data),
    )
    batch_per_epoch = batch_per_epoch or len(train_loader)

    cls           = cfg.optimizer.get("class", "Adam")
    optimizer_cfg = {k: v for k, v in cfg.optimizer.items() if k != "class"}
    optimizer     = getattr(optim, cls)(model.parameters(), **optimizer_cfg)

    if rank == 0:
        logger.warning(line)
        logger.warning(
            f"Number of parameters: {sum(p.numel() for p in model.parameters())}"
        )

    parallel_model = (
        nn.parallel.DistributedDataParallel(model, device_ids=[device])
        if world_size > 1 else model
    )

    step        = math.ceil(cfg.train.num_epoch / 10)
    best_result = float("-inf")
    best_epoch  = -1
    batch_id    = 0

    for i in range(0, cfg.train.num_epoch, step):
        parallel_model.train()
        for epoch in range(i, min(cfg.train.num_epoch, i + step)):
            if rank == 0:
                logger.warning(separator)
                logger.warning("Epoch %d begin" % epoch)

            losses = []
            sampler.set_epoch(epoch)
            for batch in islice(train_loader, batch_per_epoch):
                train_graph, batch = batch
                batch = tasks.negative_sampling(
                    train_graph, batch,
                    cfg.task.num_negative,
                    strict=cfg.task.strict_negative,
                )
                pred   = parallel_model(train_graph, batch)
                target = torch.zeros_like(pred)
                target[:, 0] = 1
                loss = F.binary_cross_entropy_with_logits(
                    pred, target, reduction="none"
                )
                neg_weight = torch.ones_like(pred)
                if cfg.task.adversarial_temperature > 0:
                    with torch.no_grad():
                        neg_weight[:, 1:] = F.softmax(
                            pred[:, 1:] / cfg.task.adversarial_temperature,
                            dim=-1,
                        )
                else:
                    neg_weight[:, 1:] = 1 / cfg.task.num_negative
                loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
                loss = loss.mean()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if rank == 0 and batch_id % cfg.train.log_interval == 0:
                    logger.warning(separator)
                    logger.warning("binary cross entropy: %g" % loss)
                losses.append(loss.item())
                batch_id += 1

            if rank == 0:
                logger.warning(separator)
                logger.warning("Epoch %d end" % epoch)
                logger.warning(line)
                logger.warning(
                    "average binary cross entropy: %g"
                    % (sum(losses) / len(losses))
                )

        epoch = min(cfg.train.num_epoch, i + step)
        if rank == 0:
            logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
            torch.save(
                {
                    "model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                save_prefix + "model_epoch_%d.pth" % epoch,
            )
        util.synchronize()

        if rank == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")
        result = test(cfg, model, valid_data, filtered_data=filtered_data)
        if result > best_result:
            best_result = result
            best_epoch  = epoch

    if rank == 0:
        logger.warning("Load checkpoint from model_epoch_%d.pth" % best_epoch)
    state = torch.load(
        save_prefix + "model_epoch_%d.pth" % best_epoch, map_location=device
    )
    model.load_state_dict(state["model"])
    util.synchronize()


@torch.no_grad()
def test(cfg, model, test_data, filtered_data=None):
    world_size  = util.get_world_size()
    rank        = util.get_rank()
    device      = util.get_device(cfg)
    all_metrics = []

    for test_graph, filters in zip(test_data, filtered_data):
        test_triplets = torch.cat([
            test_graph.target_edge_index,
            test_graph.target_edge_type.unsqueeze(0),
        ]).t()
        sampler     = torch_data.DistributedSampler(test_triplets, world_size, rank)
        test_loader = torch_data.DataLoader(
            test_triplets, cfg.train.batch_size, sampler=sampler
        )

        model.eval()
        rankings, num_negatives = [], []

        for batch in test_loader:
            t_batch, h_batch = tasks.all_negative(test_graph, batch)
            t_pred = model(test_graph, t_batch)
            h_pred = model(test_graph, h_batch)

            if filtered_data is None:
                t_mask, h_mask = tasks.strict_negative_mask(test_graph, batch)
            else:
                t_mask, h_mask = tasks.strict_negative_mask(filters, batch)

            pos_h_index, pos_t_index, _ = batch.t()
            t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
            h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
            rankings      += [t_ranking, h_ranking]
            num_negatives += [t_mask.sum(dim=-1), h_mask.sum(dim=-1)]

        ranking      = torch.cat(rankings)
        num_negative = torch.cat(num_negatives)
        all_size     = torch.zeros(world_size, dtype=torch.long, device=device)
        all_size[rank] = len(ranking)
        if world_size > 1:
            dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
        cum_size         = all_size.cumsum(0)
        all_ranking      = torch.zeros(
            all_size.sum(), dtype=torch.long, device=device
        )
        all_num_negative = torch.zeros(
            all_size.sum(), dtype=torch.long, device=device
        )
        all_ranking[
            cum_size[rank] - all_size[rank]: cum_size[rank]
        ] = ranking
        all_num_negative[
            cum_size[rank] - all_size[rank]: cum_size[rank]
        ] = num_negative
        if world_size > 1:
            dist.all_reduce(all_ranking,      op=dist.ReduceOp.SUM)
            dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)

        if rank == 0:
            for metric in cfg.task.metric:
                if metric == "mr":
                    score = all_ranking.float().mean()
                elif metric == "mrr":
                    score = (1 / all_ranking.float()).mean()
                elif metric.startswith("hits@"):
                    values    = metric[5:].split("_")
                    threshold = int(values[0])
                    if len(values) > 1:
                        num_sample = int(values[1])
                        fp_rate = (all_ranking - 1).float() / all_num_negative
                        score   = 0
                        for k in range(threshold):
                            nc = (
                                math.factorial(num_sample - 1)
                                / math.factorial(k)
                                / math.factorial(num_sample - k - 1)
                            )
                            score += (
                                nc
                                * (fp_rate ** k)
                                * ((1 - fp_rate) ** (num_sample - k - 1))
                            )
                        score = score.mean()
                    else:
                        score = (all_ranking <= threshold).float().mean()
                logger.warning("%s: %g" % (metric, score))

        mrr   = (1 / all_ranking.float()).mean()
        hits1 = (all_ranking <= 1).float().mean().item()
        combined_score = (mrr * hits1) ** 0.5

        all_metrics.append(combined_score)

        if rank == 0:
            logger.warning(
                f"MRR: {mrr:.4f} | Hits@1: {hits1:.4f} | Combined: {combined_score:.4f}"
            )
            logger.warning(separator)

    return sum(all_metrics) / len(all_metrics)


# ══════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg         = util.load_config(args.config, context=vars)

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(cfg.dataset.root):
        cfg.dataset.root = os.path.join(project_dir, cfg.dataset.root)
    if not os.path.isabs(args.ckpt):
        args.ckpt = os.path.join(project_dir, args.ckpt)

    working_dir = util.create_working_directory(cfg)

    seed_base = args.seed + util.get_rank() + 24
    os.environ["PYTHONHASHSEED"] = str(seed_base)
    random.seed(seed_base)
    np.random.seed(seed_base)
    torch.manual_seed(seed_base)
    torch.cuda.manual_seed(seed_base)
    torch.cuda.manual_seed_all(seed_base)
    torch.backends.cudnn.benchmark     = False
    torch.backends.cudnn.deterministic = True

    logger = util.get_root_logger()

    pretrain_ckpt = os.path.abspath(os.path.expanduser(args.ckpt))

    model, best_score, best_filtered_data = self_evolution_train(
        cfg, args,
        pretrain_ckpt=pretrain_ckpt,
        max_iterations=args.max_iterations,
        patience=args.patience,
        # 单关系对称
        sym_ratio_threshold=args.sym_ratio_threshold,
        sym_score_threshold=args.sym_score_threshold,
        # 关系共现
        cooccur_ratio_threshold=args.cooccur_ratio_threshold,
        min_cooccur_support=args.min_cooccur_support,
        cooccur_score_threshold=args.cooccur_score_threshold,
    )

    dataset    = util.build_dataset(cfg)
    device     = util.get_device(cfg)
    _, valid_data, test_data = dataset._data
    valid_data = [vd.to(device) for vd in valid_data]
    test_data  = [td.to(device) for td in test_data]

    test(cfg, model, valid_data, filtered_data=best_filtered_data)
    test(cfg, model, test_data,  filtered_data=best_filtered_data)
