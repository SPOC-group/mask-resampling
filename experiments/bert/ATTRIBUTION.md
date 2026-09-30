# Attribution

The training implementation is based on [DinkyTrain](https://github.com/princeton-nlp/DinkyTrain), pinned to public commit `67c5b37082f152681148c91f29390b2cd5a71170`, accompanying Alexander Wettig, Tianyu Gao, Zexuan Zhong and Danqi Chen, **Should You Mask 15% in Masked Language Modeling?** (EACL 2023). DinkyTrain builds on [fairseq](https://github.com/facebookresearch/fairseq), Myle Ott et al., **fairseq: A Fast, Extensible Toolkit for Sequence Modeling** (NAACL 2019 demonstrations), and its RoBERTa implementation. The upstream MIT license and copyright notices are retained.

The model and evaluation context are BERT (Devlin et al., 2019), RoBERTa (Liu et al., 2019), and GLUE (Wang et al., 2019). DinkyTrain's efficient pretraining recipe builds on Peter Izsak, Moshe Berchansky and Omer Levy, **How to Train BERT with an Academic Budget** (EMNLP 2021). See the manuscript bibliography and the pinned upstream README for citations and dataset provenance.

The supplied overlay retains the experiment's fixed-sequence/mask-view dataset, reproducible dynamic masks, BERT-Medium architecture registration, MLM diagnostics, evaluation metadata and tied-rank Spearman calculation, together with the Python 3.11/OmegaConf compatibility changes used for the runs. These files are unchanged from the experiment source. New portable wrappers select only the final paper configurations, disable external tracking and collect scalar results. Machine-specific launch scripts, unsuccessful recipe sweeps, private run logs, datasets and model weights are excluded.

The two corpus downloads linked in the README are public upstream resources, not project-specific mirrors. Their access conditions and licenses remain separate from this code's MIT license.
