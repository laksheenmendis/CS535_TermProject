
import torch

def calc_size(img_size, no_of_convs, conv_kernel, conv_stride, max_pool_kernel, max_pool_stride):
    initial = img_size
    for i in range(no_of_convs):
        s1 = (initial - conv_kernel)/conv_stride + 1
        m1 = (s1 - max_pool_kernel)/max_pool_stride + 1
        initial = m1
    return initial

size = calc_size(224,2,5,1,2,2)
print(size)

t1 = torch.tensor([1, 0, 0])
t2 = torch.tensor([1, 0, 0])

x = (t1 == t2).sum().float() / len(y_true)
print(x)