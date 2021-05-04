### Some modifications on excellent work by https://github.com/saic-vul/fbrs_interactive_segmentation/

### See the original repo for the requirements and setup
## Added Features:
1) Added Selection of different classes
2) Save outputs to json file on keystroke ctrl+s
3) The output json file can be used directly to train detectron2
<!--
  <p align="center">
    <img src="https://user-images.githubusercontent.com/22799415/115459081-ea66e180-a226-11eb-88bb-4d1f98916c0a.gif" alt="pruning",img width="405" />
  </p>
   <p align="center">
  -->

Example usage:

__python demo.py --gpu 0 --checkpoint  hrnet32_ocr128_lvis.pth__ 

```
@article{fbrs2020,
  title={f-BRS: Rethinking Backpropagating Refinement for Interactive Segmentation},
  author={Konstantin Sofiiuk, Ilia Petrov, Olga Barinova, Anton Konushin},
  journal={arXiv preprint arXiv:2001.10331},
  year={2020}
}
```
