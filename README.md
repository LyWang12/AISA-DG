# AISA-DG: Automatic Implicit Style-Augmented Domain Generalization on Optic Disc/Cup Segmentation

Code release for "AISA-DG: Automatic Implicit Style-Augmented Domain Generalization on Optic Disc/Cup Segmentation" (ISBI 2024)

## Paper

<div align=center><img src="hhttps://github.com/LyWang12/AISA-DG/blob/main/Figure/Fig_AISA.png" width="100%"></div>

[AISA-DG: Automatic Implicit Style-Augmented Domain Generalization on Optic Disc/Cup Segmentation](https://ieeexplore.ieee.org/abstract/document/10635681) 
(ISBI 2024)

We propose AISA-DG, an innovative style-based domain generalization framework that addresses medical image domain gaps by utilizing a novel implicit style-augmented module to enhance style diversity and model adaptability across diverse clinical segmentation datasets.


## Datasets

| Domain No. | Datasets | Train + Test | Scanners | Processed Image     |
|:-----------|:---------|:-------------|:---------|:--------------------|
| 1 | [Drishti-GS](https://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php) | 50 + 51 | (Aravind eye hospital) | [Download](https://drive.google.com/file/d/1JiJDDH1uXLu4_pKCEBmL2qbry9fH-y97/view?usp=drive_link) |
| 2 | [RIM-ONE-r3](https://rimone.isaatc.ull.es/) | 99 + 60 | Nidek AFC-210 | [Download](https://drive.google.com/file/d/1lsd8wHtKt1EbPXkW8OQcYZypkP8HxrGP/view?usp=drive_link) |
| 3 | [REFUGE (train)](https://refuge.grand-challenge.org/Home2020/) | 320 + 80 | Zeiss Visucam 500 | [Download](https://drive.google.com/file/d/1ib1YXY9yWhFWl1PawldEPcieRtijIk1p/view?usp=drive_link) |
| 4 | [REFUGE (train)](https://refuge.grand-challenge.org/Home2020/) | 320 + 80 | Canon CR-2 | [Download](https://drive.google.com/file/d/1rtqr7mRgUDMJuTmK1WU7Kh10NkWMsG2B/view?usp=drive_link) |




## Experimental Results

<div align=center><img src="https://github.com/LyWang12/AISA-DG/blob/main/Figure/Fig_Exp.png" width="100%"></div>


## Running the code

```
python train.py
```

## Testing and Visualization

```
python test.py
```


## Citation
If you find this code useful for your research, please cite our [paper](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Model_Barrier_A_Compact_Un-Transferable_Isolation_Domain_for_Model_Intellectual_CVPR_2023_paper.html):
```
@inproceedings{wang2024aisa,
  title={AISA-DG: Automatic Implicit Style-Augmented Domain Generalization on Optic Disc/Cup Segmentation},
  author={Wang, Lianyu and Fan, Dingwei and Wang, Meng and Zhang, Daoqiang},
  booktitle={2024 IEEE International Symposium on Biomedical Imaging (ISBI)},
  pages={1--4},
  year={2024},
  organization={IEEE}
}
```

## Contact
If you have any problem about our code, feel free to contact
- lywang12@126.com

or describe your problem in Issues.
