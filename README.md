# multimodal transformer for sarcasm detection in film

Since I have limited access to CPU/GPU resources, I run my code mostly on Google Colab. 
As this is a work in progess I currently run the code on separate Colab pages, but plan on having **one** Colab page for the whole model.

In addition I am trying to implement a runable local version, with only py-files.

# Colab pages:
- <What the Colab is used for<hlblv>> [<corresponding path in code<hlblv>>]() **[<comment<hlblv>>]**
- visualizing Sarcasm Headlines dataset & pre-processing [data>headline.ipynb](https://colab.research.google.com/drive/1l3D9idJvCxKm3zSOcRqBSTNL5b4Q_1wb?usp=sharing)
**[I trained two version for 2 epoches to see if the acc/f1/loss in-/decreases if working with vanilla or pre-processed dataset]**
- fine-tuning DistilBERT text-classifier [models>textclassifier.ipynb](https://colab.research.google.com/drive/1eSQKiT72yV0JJ83-_52VBaj70JU7lzFS?usp=sharing)
**[I trained two version for 2 epoches to see if the acc/f1/loss in-/decreases if working with vanilla or pre-processed dataset]**