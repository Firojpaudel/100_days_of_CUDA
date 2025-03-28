## Summary of Day 60:

> [!Note]
>**Exams are over:** Will start diving more into CUDA stuff tomorrow onwards. Today I'm just going light on README.

Okay, so today I'll try to implement the Multi-Head Self-Attention in Triton.

> [Click Here](./MHA.py) to redirect to code.

> [!important]
> ***Current Output:***
> ``` pwsh
> Input: 
>  tensor([[ 0.1940,  2.1614, -0.1721,  0.8491, -1.9244,  0.6530, -0.6494, -0.8175,
>           0.5280, -1.2753, -1.6621, -0.3033, -0.0926,  0.1992, -1.1204,  1.8577],
>         [-0.7145,  0.6881,  0.7968, -0.0334,  1.4917, -0.5165, -0.2541,  1.4746,
>          -0.3260, -1.1600,  2.3551, -0.6924,  0.1837, -1.1835, -1.8029, -1.5808],
>         [ 0.8387,  1.4192,  0.6469,  0.4253, -1.5892,  0.6223,  1.6898, -0.6648,
>           0.9425,  0.0783,  0.0847, -0.1408,  0.3316, -0.5890, -1.0723,  0.0954],
>         [-0.3347, -0.5258, -0.8776,  0.3938,  0.1640, -0.1977,  1.0104, -1.3482,
>          -0.3498, -0.6443,  0.4468, -0.5371,  1.2423, -0.8146,  0.2502, -0.4273],
>         [ 1.1044, -1.1028,  0.5543, -1.2847, -0.3816,  0.5139,  0.1002,  0.2586,
>           0.3617,  2.2787,  0.0233,  1.5828, -1.1592,  0.9484, -0.4573,  0.7605],
>         [-0.5787, -0.7050, -0.7234, -0.5071, -0.4398, -0.4182,  0.1741,  0.4427,
>           0.5069, -1.2168, -0.2719,  0.2765, -1.4398, -0.6463,  0.0749,  0.1939],
>         [ 0.5960,  0.2322,  1.1415, -0.6817, -1.6531,  0.0060,  1.3815,  1.2704,
>           0.0232, -1.3001, -0.7509,  0.3756, -0.5474, -0.0396, -0.7779, -2.5019],
>         [ 0.7000, -0.0938, -0.2163,  0.4484, -0.3152,  0.0216,  0.6253,  0.2466,
>           0.7486, -0.1169, -0.1022, -0.5011, -0.5049, -1.2072, -0.2438, -0.6784]],
>        device='cuda:0')
> Triton Output:
>  tensor([[ 0.0647,  0.0753,  0.0389,  0.0564,  0.0550, -0.1825,  0.2102,  0.0396,
>          -0.2068, -0.0304, -0.0874, -0.1118, -0.2360, -0.1059, -0.3209,  0.0039],
>         [ 0.1896,  0.0641,  0.1149, -0.0537, -0.0350, -0.1626,  0.0142, -0.0527,
>          -0.1180,  0.0693, -0.0475, -0.1715, -0.1203, -0.0402, -0.1856, -0.0497],
>         [ 0.0883,  0.1403,  0.0696,  0.0026,  0.0228, -0.1948,  0.1491, -0.0427,
>          -0.1325, -0.0311, -0.0719, -0.1915, -0.1926, -0.1006, -0.2718, -0.0031],
>         [ 0.1048,  0.0589,  0.0615, -0.0681, -0.0617, -0.0983, -0.0195, -0.0251,
>          -0.1094,  0.0198,  0.0131, -0.1684, -0.0701, -0.0411, -0.1183,  0.0142],
>         [ 0.0420,  0.0858,  0.0347, -0.0510, -0.0197, -0.1046,  0.0386, -0.0412,
>          -0.0975, -0.0443, -0.0183, -0.1616, -0.1371, -0.0660, -0.1295,  0.0382],
>         [ 0.0768,  0.0124,  0.0460,  0.0006, -0.0143, -0.0686,  0.0376,  0.0119,
>          -0.1310, -0.0100,  0.0177, -0.1030, -0.1344, -0.0746, -0.1531,  0.0286],
>         [ 0.0705,  0.0806,  0.1222, -0.0262, -0.0088, -0.1381,  0.1816, -0.0740,
>          -0.0663, -0.0037, -0.0932, -0.1328, -0.2347, -0.1798, -0.3864, -0.0602],
>         [ 0.0573,  0.0627,  0.0644, -0.0054,  0.0138, -0.1008,  0.1073, -0.0237,
>          -0.1119, -0.0103, -0.0227, -0.1254, -0.1865, -0.1235, -0.2506,  0.0057]],
>        device='cuda:0', grad_fn=<SelectBackward0>)
> PyTorch Output:
>  tensor([[ 0.0647,  0.0753,  0.0389,  0.0564,  0.0550, -0.1825,  0.2102,  0.0396,
>          -0.2068, -0.0304, -0.0874, -0.1118, -0.2360, -0.1059, -0.3209,  0.0039],
>         [ 0.1896,  0.0641,  0.1149, -0.0537, -0.0350, -0.1626,  0.0142, -0.0527,
>          -0.1180,  0.0693, -0.0475, -0.1715, -0.1203, -0.0402, -0.1856, -0.0497],
>         [ 0.0883,  0.1403,  0.0696,  0.0026,  0.0228, -0.1948,  0.1491, -0.0427,
>          -0.1325, -0.0311, -0.0719, -0.1915, -0.1926, -0.1006, -0.2718, -0.0031],
>         [ 0.1048,  0.0589,  0.0615, -0.0681, -0.0617, -0.0983, -0.0195, -0.0251,
>          -0.1094,  0.0198,  0.0131, -0.1684, -0.0701, -0.0411, -0.1183,  0.0142],
>         [ 0.0420,  0.0858,  0.0347, -0.0510, -0.0197, -0.1046,  0.0386, -0.0412,
>          -0.0975, -0.0443, -0.0183, -0.1616, -0.1371, -0.0660, -0.1295,  0.0382],
>         [ 0.0768,  0.0124,  0.0460,  0.0006, -0.0143, -0.0686,  0.0376,  0.0119,
>          -0.1310, -0.0100,  0.0177, -0.1030, -0.1344, -0.0746, -0.1531,  0.0286],
>         [ 0.0705,  0.0806,  0.1222, -0.0262, -0.0088, -0.1381,  0.1816, -0.0740,
>          -0.0663, -0.0037, -0.0932, -0.1328, -0.2347, -0.1798, -0.3864, -0.0602],
>         [ 0.0573,  0.0627,  0.0644, -0.0054,  0.0138, -0.1008,  0.1073, -0.0237,
>          -0.1119, -0.0103, -0.0227, -0.1254, -0.1865, -0.1235, -0.2506,  0.0057]],
>        device='cuda:0', grad_fn=<SelectBackward0>)
> Max difference:  tensor(1.1921e-07, device='cuda:0', grad_fn=<MaxBackward1>)
> Triton average time: 0.2431 ms
> PyTorch average time: 0.2927 ms
> Speedup (PyTorch/Triton): 1.20x
> ```