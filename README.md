# Structure-aware Video Style Transfer with Map Art
by [Thi-Ngoc-Hanh Le](https://lehanhcs.github.io/), Ya-Hsuan Chen, and [Tong-Yee Lee](https://scholar.google.com/citations?user=V3PTB98AAAAJ&hl=en&oi=ao). National Cheng-Kung University. <br>
Training code will be updated.

Paper
---
* Published online on ACM TOMM, [link](https://dl.acm.org/doi/full/10.1145/3572030)
* Project website: [link](http://graphics.csie.ncku.edu.tw/MArtVi/)
* Paper [pdf](ACM_TOMM_MapArt_Animation.pdf)

Introduction
---
This resposity is the official implementation of our method MAViNet. This paper has been published on ACM Transactions on Multimedia Computing, Communications, and Applications. <br>
Unlike the realm of transferring oil painting style, our MAViNet solves the problems of transferring map art styles to video.
![teaser_git](https://github.com/LeHanhcs/MapArt_animation/assets/37010753/dc71667a-1804-4bfc-bbfe-619ac334cb7e)


Guidance of data/code usage
---
* Folder 'data': involves digital map, video content, m-Vi video
* Folder 'weight': all trained models used in this work.
* Folder 'style': consists of map art styles
* pasteOnmap.py: takes content video and a map to produce so-called m-Vi
* infer.py --mode video --source ./data/withmap/ --model ./weight/3390454_orig_epoch_0_0630_test.pth --outputpath ./output/

Acknowledgments
---
This code benefits from [ReCoNet: Real-time Coherent Video Style Transfer Network](https://github.com/changgyhub/reconet). <br>
Our method/code is based on this code and improve it for our map art applications

Citation
---
If our method is useful for your research, please consider citing:
```
@article{le2023structure,
  title={Structure-aware Video Style Transfer with Map Art},
  author={Le, Thi-Ngoc-Hanh and Chen, Ya-Hsuan and Lee, Tong-Yee},
  journal={ACM Transactions on Multimedia Computing, Communications and Applications},
  volume={19},
  number={3s},
  pages={1--25},
  year={2023},
  publisher={ACM New York, NY}
}
```

Contact
---
If you have any question, please email me: ngochanh.le1987@gmail.com or tonylee@mail.ncku.edu.tw (corresponding author)
