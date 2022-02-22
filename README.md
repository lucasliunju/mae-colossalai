
[![logo](Colossal-AI_logo.png)](https://www.colossalai.org/)


## Masked Autoencoders: A Colossal-AI Implementation

<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/146857310-f258c86c-fde6-48e8-9cee-badd2b21bd2c.png" width="480">
</p>


This is a ColossalAI (based on PyTorch/GPU) re-implementation of the paper [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377):
```
@Article{MaskedAutoencoders2021,
  author  = {Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Doll{\'a}r and Ross Girshick},
  journal = {arXiv:2111.06377},
  title   = {Masked Autoencoders Are Scalable Vision Learners},
  year    = {2021},
}
```


* This repo is based on ColossalAI and you can install it:
```
pip install colossalai
```
* This repo is also based on the [MAE-Pytorch](https://github.com/facebookresearch/mae) official implementation. Installation and preparation follow that repo.

* This repo is based on [`timm==0.3.2`](https://github.com/rwightman/pytorch-image-models), for which a [fix](https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842) is needed to work with PyTorch 1.8.1+.

### Pre-training

To run multi-GPU training on a single node, you can directly run `run.sh`.

```
bash run.sh
```


### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
