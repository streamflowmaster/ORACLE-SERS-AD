import numpy as np
import pandas as pd
import json

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
    "multiply": lambda xs: np.prod(xs, axis=0),
    "segmented": lambda xs: xs[1] * xs[2] if xs[0] > 2 else 0,
    "mixed": lambda xs: xs[0] * xs[1] ** 2 * xs[2],
    "nonlinear": lambda xs: np.sin(np.prod(xs, axis=0))
}


def generate_from_settings(settings_file):
    # 读取设定
    with open(settings_file, "r") as f:
        settings = json.load(f)

    # 设置随机种子
    np.random.seed(settings["random_seed"])

    n_samples = settings["n_samples"]
    n_classes = settings["n_classes"]
    var_names = [v for v in settings["variable_distributions"] if v != "Z"] + ["Z"]

    # 生成变量
    data = {}
    for var, dist in settings["variable_distributions"].items():
        if dist["type"] == "continuous":
            if dist["params"].get("low") is not None:
                data[var] = np.random.uniform(dist["params"]["low"], dist["params"]["high"], n_samples)
            else:
                data[var] = np.random.normal(dist["params"]["mean"], dist["params"]["std"], n_samples)
        elif dist["type"] == "discrete":
            data[var] = np.random.choice(dist["levels"], size=n_samples, p=dist["probs"])
        elif dist["type"] == "confounder":
            if dist.get("weights"):
                Z = sum(w * data[p] for w, p in zip(dist["weights"], dist["parents"])) + np.random.normal(0, 0.1,
                                                                                                          n_samples)
            else:
                Z = np.array([np.random.choice(dist["levels"], p=dist["probs_table"][i % len(dist["parents"])]) for i in
                              range(n_samples)])
            data[var] = Z

    # 计算潜在函数
    for k in range(n_classes):
        f_k = np.zeros(n_samples)
        for term in settings["latent_functions"][k]["terms"]:
            if "variables" in term:
                xs = np.vstack([data[v] for v in term["variables"]]).T
                f_k += term["weight"] * interaction_types[term["transform"]](xs)
            else:
                f_k += term["weight"] * transforms[term["transform"]](data[term["variable"]])
        f_k += np.random.normal(0, 0.1, n_samples)
        f_k += term["bias"]
        data[f"f{k}"] = f_k

    # 计算 softmax 概率
    scores = np.vstack([data[f"f{k}"] for k in range(n_classes)]).T
    exp_scores = np.exp(scores)
    P = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # 生成 Y
    Y = np.array([np.random.choice(range(n_classes), p=p) for p in P])
    data["Y"] = Y

    # 保存数据
    data_df = pd.DataFrame({k: data[k] for k in var_names + ["Y"]})
    dataset_id = settings["dataset_id"]
    data_df.to_csv(f"datasets/{dataset_id}_reproduced_data.csv", index=False)

    print(f"重现数据集 {dataset_id}，类别分布：")
    print(data_df["Y"].value_counts(normalize=True))


# 示例：重现数据集

# generate_from_settings("datasets/ds_001_settings.json")