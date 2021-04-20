## Some modifications on excellent work based on https://github.com/saic-vul/fbrs_interactive_segmentation/
### See the original repo for the requirements and setup
## Added Features:
1) Added Selection of different classes
2) Save outputs to json file on keystroke ctrl+s
3) The output json file can be used directly to train detectron2

Example usage:

python demo.py --gpu 0 --checkpoint  hrnet32_ocr128_lvis.pth 


@article{fbrs2020,
  title={f-BRS: Rethinking Backpropagating Refinement for Interactive Segmentation},
  author={Konstantin Sofiiuk, Ilia Petrov, Olga Barinova, Anton Konushin},
  journal={arXiv preprint arXiv:2001.10331},
  year={2020}
}
```
