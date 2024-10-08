# TS-STDN (ICONIP 2023)

![TS-STDN model](TS-STDNmodel.png)

This is the official repository of our paper for ICONIP 23: [Two-Stream Spectral-Temporal Denoising Network for End-to-End Robust EEG-Based Emotion Recognition](https://link.springer.com/chapter/10.1007/978-981-99-8067-3_14).

- **Abstraction**: Emotion recognition based on electroencephalography (EEG) is attracting more and more interest in affective computing. Previous studies have predominantly relied on manually extracted features from EEG signals. It remains largely unexplored in the utilization of raw EEG signals, which contain more temporal information but present a significant challenge due to their abundance of redundant data and susceptibility to contamination from other physiological signals, such as electrooculography (EOG) and electromyography (EMG). To cope with the high dimensionality and noise interference in end-to-end EEG-based emotion recognition tasks, we introduce a Two-Stream Spectral-Temporal Denois- ing Network (TS-STDN) which takes into account the spectral and temporal aspects of EEG signals. Moreover, two U-net modules are adopted to reconstruct clean EEG signals in both spectral and temporal domains while extracting discriminative features from noisy data for classifying emotions. Extensive experiments are conducted on two public datasets, SEED and SEED-IV, with the original EEG signals and the noisy EEG signals contaminated by EMG signals. Compared to the baselines, our TS-STDN model exhibits a notable improvement in accuracy, demonstrating an increase of 6% and 8% on the clean data and 11% and 10% on the noisy data, which shows the robustness of the model.

The source code of the TS-STDN model and producing noisy data.

## Citation
If you find our paper/code/dataset useful, please consider citing our work:
```
@inproceedings{liu2023two,
  title={Two-Stream Spectral-Temporal Denoising Network for End-to-End Robust EEG-Based Emotion Recognition},
  author={Liu, Xuan-Hao and Jiang, Wei-Bang and Zheng, Wei-Long and Lu, Bao-Liang},
  booktitle={International Conference on Neural Information Processing},
  pages={186--197},
  year={2023},
  organization={Springer}
}
```
