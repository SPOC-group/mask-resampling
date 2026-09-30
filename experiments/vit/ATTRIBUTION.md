# Source attribution

This implementation adapts **U-MAE**, the code accompanying Qi Zhang, Yifei Wang and Yisen Wang, *How Mask Matters: Towards Theoretical Understandings of Masked Autoencoders* (NeurIPS 2022):
https://github.com/zhangq327/U-MAE

U-MAE builds on **MAE**, Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar and Ross Girshick, *Masked Autoencoders Are Scalable Vision Learners* (CVPR 2022):
https://github.com/facebookresearch/mae

The inherited copyright notices and MAE's Creative Commons Attribution-NonCommercial 4.0 license are retained in the source and `LICENSE`. References to additional upstream implementations remain in their original headers. Dependencies retain their own licenses.

The experiment code adds fixed per-image masking, deterministic preprocessing, unmasked reconstruction, shared initialization and checkpoint controls. Supplementary packaging adds portable presets, result collection and the exact class list, removes external experiment tracking and private orchestration, and suppresses Git revision recording. The numerical model, loss, optimizer and training-loop code is retained. Uniformity regularization is disabled in all supplied paper presets. The longer schedules are adaptations of U-MAE/MAE optimization recipes, not published checkpoints or exact reproductions of their published experiments.

```bibtex
@inproceedings{zhang2022how,
  title={How Mask Matters: Towards Theoretical Understandings of Masked Autoencoders},
  author={Zhang, Qi and Wang, Yifei and Wang, Yisen},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
@inproceedings{he2022masked,
  title={Masked Autoencoders Are Scalable Vision Learners},
  author={He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and Doll{\'a}r, Piotr and Girshick, Ross},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```
