# TS-STDN: A Network for Robust End-to-end EEG-based Emotion Recognition
---

ChatArena is a library that provides multi-agent language game environments and facilitates research about autonomous
LLM agents and their social interactions.
It provides the following features:

- **Abstraction**: Abstract. Emotion recognition based on electroencephalography (EEG)
is attracting more and more interest in affective computing. Previous
studies have predominantly relied on manually extracted features from
EEG signals. It remains largely unexplored in the utilization of raw EEG
signals, which contain more temporal information but present a significant challenge due to their abundance of redundant data and susceptibility to contamination from other physiological signals, such as electrooculography (EOG) and electromyography (EMG). To cope with the high
dimensionality and noise interference in end-to-end EEG-based emotion
recognition tasks, we introduce a Two-Stream Spectral-Temporal Denoising Network (TS-STDN) which takes into account the spectral and temporal aspects of EEG signals. Moreover, two U-net modules are adopted
to reconstruct clean EEG signals in both spectral and temporal domains
while extracting discriminative features from noisy data for classifying
emotions. Extensive experiments are conducted on two public datasets,
SEED and SEED-IV, with the original EEG signals and the noisy EEG
signals contaminated by EMG signals. Compared to the baselines, our
TS-STDN model exhibits a notable improvement in accuracy, demonstrating an increase of 6% and 8% on the clean data and 11% and 10%
on the noisy data, which shows the robustness of the model.

![ChatArena Architecture](modelbig.pdf)

The source code of TS-STDN model
