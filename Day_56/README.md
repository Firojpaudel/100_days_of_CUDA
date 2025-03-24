## Summary of Day 56:

> *Exam week so doing something not that heavy.

So today, I'll try to implement a ReLU in triton:

> [Click Here](./relu.py) to redirect towards code.

> ***Output Check***:
>```bash
> Input was:  tensor([ 0.4282,  0.3000,  2.2080,  0.3899, -0.3662,  1.4428,  0.1740,  0.2815,
>          0.6397,  1.5522], device='cuda:0')
> Output is:  tensor([0.4282, 0.3000, 2.2080, 0.3899, 0.0000, 1.4428, 0.1740, 0.2815, 0.6397,
>         1.5522], device='cuda:0')
> PyTorch Expected Values: tensor([0.4282, 0.3000, 2.2080, 0.3899, 0.0000, 1.4428, 0.1740, 0.2815, 0.6397,
>         1.5522], device='cuda:0')
>```