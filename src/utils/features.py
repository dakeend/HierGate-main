from Bio.PDB import *
from Bio.PDB.Polypeptide import PPBuilder


def three_to_one(residue):
    """Convert three-letter amino acid code to one-letter code."""
    d = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    return d.get(residue.upper(), 'X')


def read_pssm_file(pssm_path):
    """
    读取PSSM文件并提取位置特异性得分矩阵特征

    Args:
        pssm_path: PSSM文件路径

    Returns:
        profile: 包含每个位置的PSSM得分的列表，每个元素是长度为20的列表
    """
    profile = []

    with open(pssm_path, "r") as f:
        for line in f:
            line = line.strip()

            # 跳过空行和注释行
            if not line or line.startswith("Last position") or line.startswith("            A"):
                continue

            # 分割行数据
            parts = line.split()

            # 检查是否是数据行（应该以数字开头，然后是氨基酸）
            if len(parts) >= 22 and parts[0].isdigit():
                try:
                    # 提取位置编号（第1列）和氨基酸（第2列）
                    position = int(parts[0])
                    aa = parts[1]

                    # 提取20个氨基酸位置的得分（第3-22列）
                    # 对应 A R N D C Q E G H I L K M F P S T W Y V
                    scores = [int(parts[i]) for i in range(2, 22)]

                    profile.append(scores)

                except (ValueError, IndexError):
                    # 如果解析失败，跳过这一行
                    continue

    return profile


def read_pssm_file_normalized(pssm_path):
    """
    读取PSSM文件并返回归一化的特征

    Args:
        pssm_path: PSSM文件路径

    Returns:
        profile: 包含每个位置的归一化PSSM得分和概率的列表
    """
    profile = []

    with open(pssm_path, "r") as f:
        for line in f:
            line = line.strip()

            # 跳过空行和注释行
            if not line or line.startswith("Last position") or line.startswith("            A"):
                continue

            # 分割行数据
            parts = line.split()

            # 检查是否是数据行
            if len(parts) >= 42 and parts[0].isdigit():
                try:
                    # 提取位置编号和氨基酸
                    position = int(parts[0])
                    aa = parts[1]

                    # 提取20个氨基酸位置的得分（第3-22列）
                    scores = [int(parts[i]) for i in range(2, 22)]

                    # 提取概率百分比（第23-42列）
                    percentages = [int(parts[i]) for i in range(22, 42)]

                    # 组合得分和概率作为特征
                    # 可以选择只用得分、只用概率，或者两者都用
                    combined_features = scores + percentages  # 40维特征
                    # 或者只用得分: features = scores  # 20维特征
                    # 或者只用概率: features = percentages  # 20维特征

                    profile.append(combined_features)

                except (ValueError, IndexError):
                    continue

    return profile


def read_next_nline(f, n):
    for i in range(n):
        line = f.readline()
    return line


def profile2freq(value):
    if value == "*":
        return 0
    else:
        return 2 ** (-int(value) / 1000)


def read_hhm_file(hhm):
    step = 1
    profile = []
    with open(hhm, "r") as f:
        line = f.readline()
        while line and not line.startswith("//"):
            if line.startswith("HMM "):
                step = 3
            line = read_next_nline(f, step)
            if step == 3 and not line.startswith("//"):
                data = [profile2freq(v) for v in line.split()[2:-1]]
                profile.append(data)

        return profile


def read_scoring_functions(pdb):
    scoring = False
    profile = []
    for line in open(pdb):
        if line.startswith("VRT"):
            scoring = False
        if scoring:
            data = [float(v) for v in line.split()[1:-1]]
            profile.append(data)
        if line.startswith("pose"):
            scoring = True
    return profile


def load_aa_features(feature_path):
    aa_features = {}
    for line in open(feature_path):
        line = line.strip().split()
        aa, features = line[0], line[1:]
        features = [float(feature) for feature in features]
        aa_features[aa] = features
    return aa_features


def get_node_feature(nodes_list, profile, scoring, aa_features, chain):
    features = []

    ppb = PPBuilder()
    pp = ppb.build_peptides(chain)
    res_list = []
    for p in pp:
        res_list.extend(p)

    for node in nodes_list:
        res = chain[int(node)]
        # if res.get_resname() not in ["HEC", "IPA"]:

        data = list(profile[res_list.index(res)])   # positional encoding
        score = list(scoring[res_list.index(res)])  # rosetta scoring function
        aa_feature = aa_features[three_to_one(res.get_resname())]   # sequence encoding
        features.append(data + score + aa_feature)

    return features


def get_node_feature_with_pssm(nodes_list, profile, scoring, aa_features, pssm_profile, chain):
    """
    获取包含PSSM特征的节点特征

    Args:
        nodes_list: 节点列表
        profile: HHM profile特征
        scoring: Rosetta评分函数特征
        aa_features: 氨基酸特征
        pssm_profile: PSSM特征
        chain: 蛋白质链

    Returns:
        features: 包含所有特征的列表
    """
    features = []

    ppb = PPBuilder()
    pp = ppb.build_peptides(chain)
    res_list = []
    for p in pp:
        res_list.extend(p)

    for node in nodes_list:
        res = chain[int(node)]
        res_index = res_list.index(res)

        # HHM profile特征
        hhm_data = list(profile[res_index]) if res_index < len(profile) else [0] * len(profile[0])

        # Rosetta评分函数特征
        score = list(scoring[res_index]) if res_index < len(scoring) else [0] * len(scoring[0])

        # 氨基酸特征
        aa_feature = aa_features[three_to_one(res.get_resname())]

        # PSSM特征
        pssm_data = list(pssm_profile[res_index]) if res_index < len(pssm_profile) else [0] * 20

        # 组合所有特征
        combined_features = hhm_data + score + aa_feature + pssm_data
        # combined_features = hhm_data + aa_feature + score
        features.append(combined_features)

    return features
    