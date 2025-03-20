## Summary of Day 52:

> *Outside of PMPP Learning Triton 

> Triton is compatible mainly on Linux. To make it run on Windows, Follow these steps:
> 1. You'll need CUDA 
> 2. You'll need PyTorch
> 3. Then install triton using command 
>       ```pwsh
>       pip install -U triton-windows
>       ```
> 4. **Test if its working**:
>       On Powershell/terminal type `python -c "import triton; print(triton.__version__)"` and it should provide a version number.

Okay now that the installation is done, let's learn some basic syntaxes!

> *[Click Here](./vect_add.py) to redirect to Code_Example1- Vector addition*
>
> ***This is how output would look like:***
> ```bash
>  Resulting Tensor Z: tensor([ 2.8879, -0.9332, -0.8542,  ..., -3.4584, -0.7213, -1.6162],
>         device='cuda:0')
>  PyTorch Result: tensor([ 2.8879, -0.9332, -0.8542,  ..., -3.4584, -0.7213, -1.6162],
>        device='cuda:0')
>  Maximum Difference: tensor(0., device='cuda:0')
> ```


> *[Click Here](./mat_mul.py) to redirect to Code_Example2- Matrix Multiplication*
>
> ***This is how output would look like:***
> ```bash 
> Triton Multiplication Output: tensor([[ -6.0559,  14.7592,   9.6860,  ...,  10.2575,  -6.2977,  -7.8479],
>         [ -8.1676, -11.9154,  -5.5921,  ..., -13.0701,   2.5830, -23.8680],
>         [ 11.5374,  15.3215,   9.8456,  ...,  -7.4482,   2.4150,   8.1242],
>         ...,
>         [ -8.5090,  24.6033, -35.0272,  ..., -15.3333,  -7.1859, -10.5077],
>         [  9.1017,   8.7194,  -1.6191,  ...,   4.5717,   2.6488,  -8.9270],
>        [ 17.0029,  -3.3823,   6.7352,  ..., -14.5266,  11.6854,  14.6934]],
>        device='cuda:0')
> PyTorch Multiplication Output: tensor([[ -6.0564,  14.7729,   9.6952,  ...,  10.2634,  -6.3079,  -7.8491],
>         [ -8.1747, -11.9333,  -5.5906,  ..., -13.0820,   2.5793, -23.8771],
>         [ 11.5403,  15.3292,   9.8522,  ...,  -7.4565,   2.4286,   8.1329],
>         ...,
>         [ -8.5142,  24.6182, -35.0537,  ..., -15.3436,  -7.1959, -10.5152],
>         [  9.1087,   8.7276,  -1.6206,  ...,   4.5772,   2.6595,  -8.9320],
>         [ 17.0106,  -3.3855,   6.7297,  ..., -14.5470,  11.6929,  14.7005]],
>        device='cuda:0')
> Maximum Difference: 0.049560546875
> ```

> [Click Here]() to redirect to Code_Example3- Softmax *
>
> ***This is how output would look like:***
> ```bash
> Input Tensor: tensor([[ 2.1458, -0.0763,  0.3136, -0.5863,  0.2722],
>        [-1.2725,  1.4601, -0.1411, -1.4648,  0.2158],
>        [ 0.0374,  0.4764,  0.4209,  0.4245,  1.6977]], device='cuda:0')
>
>Triton Softmax Output: tensor([[0.6724, 0.0729, 0.1076, 0.0438, 0.1033],
>        [0.0404, 0.6217, 0.1254, 0.0334, 0.1791],
>        [0.0930, 0.1443, 0.1365, 0.1370, 0.4893]], device='cuda:0')
>
>PyTorch Softmax Output: tensor([[0.6724, 0.0729, 0.1076, 0.0438, 0.1033],
>        [0.0404, 0.6217, 0.1254, 0.0334, 0.1791],
>        [0.0930, 0.1443, 0.1365, 0.1370, 0.4893]], device='cuda:0')
>Maximum Difference: 2.9802322387695312e-08
> ```

