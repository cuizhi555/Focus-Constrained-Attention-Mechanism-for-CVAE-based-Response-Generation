## Focus-Constrained-Attention-Mechanism-for-CVAE-based-Response-Generation
The implementation of "Focus-Constrained Attention Mechanism for CVAE-based Response Generation"  https://www.aclweb.org/anthology/2020.findings-emnlp.183.pdf


### Directory
```
data: samples of weibo conversational dataset
focusseq: the main implementation codes
run_focusseq: the running scripts
```

### requirements
```
python == 2.7
tensorflow == 1.11.0
```

### Commands

##### Step1: Preprocess
```
cd run_focusseq
bash preprocess.sh
```

##### Step2: Train
```
cd run_focusseq
bash train.sh
```

##### Step3: Test
```
cd run_focusseq
bash decode.sh
```


### Disclaimer
```
We implement our model based on the following project:  https://github.com/tensorflow/models/tree/master/official/nlp/transformer

Our project is only for research purpose only. 

Please contact Zhi Cui (allencui555@gmail.com) for any further questions.
```

### Citation

```bibtex
@inproceedings{CuiLZCWW20,
  author    = {Zhi Cui and Yanran Li and Jiayi Zhang and Jianwei Cui and Chen Wei and Bin Wang},
  title     = {Focus-Constrained Attention Mechanism for CVAE-based Response Generation},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings, {EMNLP} 2020, Online Event, 16-20 November 2020},
  year      = {2020},
}
```
