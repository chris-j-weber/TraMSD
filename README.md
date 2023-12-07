## Transformer-Based Multimodal Sarcasm Detection in Film

Sarcasm is often communicated through spoken words linked together with \textit{verbal} elements, e.g. varying tones, and \textit{visual} elements, e.g. facial expressions, all of which occur \textit{over time}. However, current state-of-the-art works for sarcasm detection either focus only on textual elements or use a static combination of textual and visual elements. The temporal context crucial in cinematic storytelling is thus often neglected. To address this issue, a \textit{multimodal} deep learning approach that considers \textit{dynamic} textual and visual information is proposed in this thesis. For this purpose, a Transformer-based Multimodal Sarcasm Detection network (TraMSD) is designed to focus on multimodal context-relevant information. TraMSD integrates pre-trained Transformer encoders for text and vision, respectively, and combines these unimodal features with a multimodal module to learn patterns indicative of sarcasm that are shared across modalities. The multimodal module comes in two flavours, one that leverages a late fusion while the other uses a cross-attention mechanism.
The proposed method is compared against a unimodal text-only baseline and a current state-of-the-art multimodal approach. Extensive experiments demonstrate that TraMSD outperforms the baselines particularly when employing the cross-attention module. The network utilises max pooling to highlight significant sentence features, and applies attention pooling to emphasise sentences that of importance for conveying sarcasm. An ablation study provides insights on the generalisability of TraMSD and explores the potential of the model in low data regimes.
Overall, TraMSD presents a novel approach to multimodal sarcasm detection in film, laying a strong foundation for future advancements in this field. 

## Dataset

The transformer approache has been trained and evaluated on the [MUStARD++](https://github.com/cfiltnlp/MUStARD_Plus_Plus) dataset, which contains sarcastic and non-sarcastic video clips from popular American sitcoms. The MUStARD++ is a reworked dataset by [Anupama Ray](https://aclanthology.org/2022.lrec-1.756.pdf) and originates from Santiago Castro's [MUStARD](https://github.com/soujanyaporia/MUStARD) dataset.

## Architecture

![TraMSD](/material/tramsd.png)

