import numpy as np
import pandas as pd
import json
import os

# 设置全局随机种子
np.random.seed(42)

# 数据集配置
dataset_configs = [
    {"n_samples": 5000, "n_variables": 10, "n_classes": 8, "discrete_ratio": 0.1, "decision_ratio": 0.5},
    {"n_samples": 5000, "n_variables": 5, "n_classes": 3, "discrete_ratio": 0.4, "decision_ratio": 0.6},
    {"n_samples": 10000, "n_variables": 7, "n_classes": 4, "discrete_ratio": 0.5, "decision_ratio": 0.8},
    {"n_samples": 5000, "n_variables": 10, "n_classes": 2, "discrete_ratio": 0.8, "decision_ratio": 0.4},
    {"n_samples": 5000, "n_variables": 15, "n_classes": 3, "discrete_ratio": 0.3, "decision_ratio": 0.3},
    {"n_samples": 50000, "n_variables": 30, "n_classes": 3, "discrete_ratio": 0.4, "decision_ratio": 0.7},
    {"n_samples": 10000, "n_variables": 7, "n_classes": 4, "discrete_ratio": 0.5, "decision_ratio": 0.2},
    {"n_samples": 5000, "n_variables": 15, "n_classes": 3, "discrete_ratio": 0.2, "decision_ratio": 0.1},
    {"n_samples": 10000, "n_variables": 12, "n_classes": 5, "discrete_ratio": 0.5, "decision_ratio": 0.9},
    {"n_samples": 5000, "n_variables": 18, "n_classes": 6, "discrete_ratio": 0.7, "decision_ratio": 0.4}

]*10

# 变换函数
transforms = {
    "linear": lambda x: x,
    "square": lambda x: x ** 2,
    "sin": lambda x: np.sin(x),
    "log": lambda x: np.log(np.abs(x) + 1),
    "discrete": lambda x: x
}

# 交互形式
interaction_types = {
    "multiply": lambda xs: np.prod(xs, axis=1),
    "segmented": lambda xs: np.where(xs[:, 0] > 2, xs[:, 1] * xs[:, 2] if xs.shape[1] > 2 else xs[:, 1], 0),
    "mixed": lambda xs: xs[:, 0] * xs[:, 1] ** 2 * (xs[:, 2] if xs.shape[1] > 2 else 1),
    "nonlinear": lambda xs: np.sin(np.prod(xs, axis=1)),
    "categorical": lambda xs: np.all(xs[:, :min(2, xs.shape[1])] > 1, axis=1).astype(float)
}


def generate_variable(n_samples, dist_type, params):
    if dist_type == "uniform":
        return np.random.uniform(params["low"], params["high"], n_samples)
    elif dist_type == "normal":
        return np.random.normal(params["mean"], params["std"], n_samples)
    elif dist_type == "discrete":
        return np.random.choice(params["levels"], size=n_samples, p=params["probs"])


def generate_dataset(config, dataset_id):
    n_samples = int(config["n_samples"])
    n_variables = int(config["n_variables"])
    n_classes = int(config["n_classes"])
    discrete_ratio = float(config["discrete_ratio"])
    decision_ratio = float(config["decision_ratio"])

    # 初始化设定
    settings = {
        "dataset_id": f"ds_{dataset_id:03d}",
        "n_samples": n_samples,
        "n_variables": n_variables,
        "n_classes": n_classes,
        "random_seed": np.random.randint(0, 10000),
        "variable_distributions": {},
        "causal_graph": [],
        "latent_functions": [],
        "decision_variables": [],
        "interaction_orders": [],
        "interaction_order_probs": []
    }

    # 设置随机种子
    np.random.seed(settings["random_seed"])

    # 确定离散和连续变量
    n_discrete = int(n_variables * discrete_ratio)
    n_decision = max(2, int(n_variables * decision_ratio))
    if n_decision > n_variables:
        n_decision = n_variables
    var_names = [f"X{i + 1}" for i in range(n_variables)] + ["Z"]
    discrete_vars = np.random.choice(var_names[:-1], size=n_discrete, replace=False)
    decision_vars = np.random.choice(var_names[:-1], size=n_decision, replace=False)
    settings["decision_variables"] = list(decision_vars)
    assert len(decision_vars) >= 2, "At least 2 decision variables required for interactions"

    # 生成变量
    data = {}
    for var in var_names[:-1]:
        if var in discrete_vars:
            max_levels = np.random.choice([2, 3, 4, 5])
            levels = list(range(1, max_levels + 1))
            alpha_options = [
                np.random.uniform(0.5, 2, len(levels)),
                np.array([5] + [1] * (len(levels) - 1))
            ]
            alpha = alpha_options[np.random.choice([0, 1], p=[0.7, 0.3])]
            probs = np.random.dirichlet(alpha)
            assert abs(sum(probs) - 1.0) < 1e-6, f"Probabilities for {var} do not sum to 1"
            settings["variable_distributions"][var] = {
                "type": "discrete",
                "levels": levels,
                "probs": probs.tolist(),
                "categorical": True
            }
            data[var] = generate_variable(n_samples, "discrete", {"levels": levels, "probs": probs})
        else:
            dist_type = np.random.choice(["uniform", "normal"])
            params = {"low": -1, "high": 1} if dist_type == "uniform" else {"mean": 0, "std": 1}
            settings["variable_distributions"][var] = {
                "type": dist_type,
                "params": params
            }
            data[var] = generate_variable(n_samples, dist_type, params)

    # 生成混淆变量 Z
    z_type = np.random.choice(["continuous", "discrete"])
    z_parents = np.random.choice(
        [v for v in var_names[:-1] if v not in decision_vars] or var_names[:-1],
        size=np.random.randint(1, min(3, n_variables)),

        # replace=False
    )
    if z_type == "continuous":
        z_weights = np.random.uniform(0.1, 0.3, len(z_parents))
        Z = sum(w * data[p] for w, p in zip(z_weights, z_parents)) + np.random.normal(0, 0.1, n_samples)
        settings["variable_distributions"]["Z"] = {
            "type": "continuous",
            "parents": list(z_parents),
            "weights": z_weights.tolist()
        }
    else:
        levels = list(range(1, np.random.choice([2, 3, 4]) + 1))
        alpha_options = [
            np.random.uniform(0.5, 2, len(levels)),
            np.array([5] + [1] * (len(levels) - 1))
        ]
        alpha = alpha_options[np.random.choice([0, 1], p=[0.7, 0.3])]
        probs_table = np.random.dirichlet(alpha, size=len(z_parents))
        Z = np.array([np.random.choice(levels, p=probs_table[i % len(z_parents)]) for i in range(n_samples)])
        settings["variable_distributions"]["Z"] = {
            "type": "discrete",
            "parents": list(z_parents),
            "levels": levels,
            "probs_table": probs_table.tolist()
        }
    data["Z"] = Z

    # 定义因果图
    settings["causal_graph"] = [{"source": var, "target": "Y"} for var in decision_vars] + [
        {"source": "Z", "target": "Y"}]
    for parent in z_parents:
        settings["causal_graph"].append({"source": parent, "target": "Z"})

    # 生成潜在函数
    n_interactions = np.random.randint(1, 4)
    max_order = min(len(decision_vars), 4)
    possible_orders = [i for i in [2, 3, 4] if i <= max_order] or [2]
    assert len(possible_orders) > 0, "Possible orders cannot be empty"
    base_probs = [0.5, 0.4, 0.1][:len(possible_orders)]
    probs = np.array(base_probs) / np.sum(base_probs) if base_probs else [1.0]
    assert abs(sum(probs) - 1.0) < 1e-6, f"Probabilities for interaction orders do not sum to 1: {probs}"
    interaction_orders = np.random.choice(possible_orders, size=n_interactions, p=probs)
    settings["interaction_orders"] = interaction_orders.tolist()
    settings["interaction_order_probs"] = probs.tolist()

    for k in range(n_classes):
        terms = []
        for var in decision_vars:
            transform = "discrete" if var in discrete_vars else np.random.choice(list(transforms.keys())[:-1])
            weight = np.random.uniform(0.1, 0.5)
            terms.append({"variable": var, "transform": transform, "weight": weight})
        for order in interaction_orders:
            interaction_type = np.random.choice(list(interaction_types.keys()))
            interaction_vars = np.random.choice(
                [v for v in decision_vars if v in discrete_vars] or decision_vars,
                size=order,
                # replace=False
            ) if interaction_type == "categorical" else np.random.choice(decision_vars, size=order, replace=False)
            weight_int = np.random.uniform(0.05, 0.2)
            terms.append({
                "variables": list(interaction_vars),
                "transform": interaction_type,
                "weight": weight_int,
                "is_discrete": all(v in discrete_vars for v in interaction_vars)
            })
        weight_z = np.random.uniform(0.05, 0.1)
        z_transform = "discrete" if z_type == "discrete" else "linear"
        terms.append({"variable": "Z", "transform": z_transform, "weight": weight_z})

        f_k = np.zeros(n_samples)
        for term in terms:
            if "variables" in term:
                xs = np.vstack([data[v] for v in term["variables"]]).T
                f_k += term["weight"] * interaction_types[term["transform"]](xs)
            else:
                f_k += term["weight"] * transforms[term["transform"]](data[term["variable"]])
        f_k += np.random.normal(0, 0.1, n_samples)

        b_k = -np.mean(f_k)
        f_k += b_k

        settings["latent_functions"].append({
            "class": k,
            "terms": terms,
            "bias": float(b_k)
        })
        data[f"f{k}"] = f_k

    # 计算 softmax 概率
    scores = np.vstack([data[f"f{k}"] for k in range(n_classes)]).T
    exp_scores = np.exp(scores)
    P = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # 生成 Y
    Y = P.argmax(axis=-1)
    # Y = np.array([np.random.choice(range(n_classes), p=p) for p in P])
    data["Y"] = Y

    # 保存数据
    data_df = pd.DataFrame({k: data[k] for k in var_names + ["Y"]})
    os.makedirs("datasets", exist_ok=True)
    data_df.to_csv(f"datasets/{settings['dataset_id']}_data.csv", index=False)

    # 保存设定
    with open(f"datasets/{settings['dataset_id']}_settings.json", "w") as f:
        json.dump(settings, f, indent=4)

    print(f"生成数据集 {settings['dataset_id']}，类别分布：")
    print(data_df["Y"].value_counts(normalize=True))


# 批量生成
for i, config in enumerate(dataset_configs):
    generate_dataset(config, i + 1)