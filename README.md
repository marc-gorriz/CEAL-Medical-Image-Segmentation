# Cost-Effective Active Learning for Melanoma Segmentation

| ![Marc Górriz][MarcGorriz-photo]  |  ![Axel Carlier][AxelCarlier-photo] | ![Emmanuel Faure][EmmanuelFaure-photo]  | ![Xavier Giro-i-Nieto][XavierGiro-photo]  |
|:-:|:-:|:-:|:-:|
| [Marc Górriz][MarcGorriz-web]  | [Axel Carlier][AxelCarlier-web] | [Emmanuel Faure][EmmanuelFaure-web] | [Xavier Giro-i-Nieto][XavierGiro-web] |

[MarcGorriz-web]: https://www.linkedin.com/in/marc-górriz-blanch-74501a123/
[XavierGiro-web]: https://imatge.upc.edu/web/people/xavier-giro
[AxelCarlier-web]: http://carlier.perso.enseeiht.fr
[EmmanuelFaure-web]: https://www.irit.fr/~Emmanuel.Faure/



[MarcGorriz-photo]: https://raw.githubusercontent.com/marc-gorriz/CEAL-Medical-Image-Segmentation/master/authors/MarcGorriz.jpg
[XavierGiro-photo]: https://raw.githubusercontent.com/marc-gorriz/CEAL-Medical-Image-Segmentation/master/authors/XavierGiro.jpg
[AxelCarlier-photo]: https://raw.githubusercontent.com/marc-gorriz/CEAL-Medical-Image-Segmentation/master/authors/AxelCarlier.jpg
[EmmanuelFaure-photo]: https://raw.githubusercontent.com/marc-gorriz/CEAL-Medical-Image-Segmentation/master/authors/EmmanuelFaure.png

A joint collaboration between:

| ![logo-vortex] | ![logo-enseeiht] | ![logo-gpi] |
|:-:|:-:|:-:|
| [IRIT Vortex Group][vortex-web] | [INP Toulouse - ENSEEIHT][enseeiht-web] | [UPC Image Processing Group][gpi-web] |

[vortex-web]: https://www.irit.fr/-VORTEX-Team-?lang=fr/
[enseeiht-web]: http://www.enseeiht.fr/fr/index.html/
[upc-web]: http://www.upc.edu/?set_language=en/
[etsetb-web]: https://www.etsetb.upc.edu/en/ 
[gpi-web]: https://imatge.upc.edu/web/ 


[logo-vortex]: https://github.com/marc-gorriz/CEAL-Medical-Image-Segmentation/blob/master/logos/Vortex.png "VORTEX Team (IRIT)"
[logo-enseeiht]: https://raw.githubusercontent.com/marc-gorriz/CEAL-Medical-Image-Segmentation/master/logos/enseeiht.png "Institut National polytechnique de Toulouse (ENSEEIHT)"
[logo-gpi]: https://raw.githubusercontent.com/marc-gorriz/CEAL-Medical-Image-Segmentation/master/logos/gpi.png "UPC GPI"

## Abstract

We propose a novel Active Learning framework capable to train effectively a convolutional neural network for semantic segmentation of medical imaging, with a limited amount of training labeled data. Our contribution is a practical Cost-Effective Active Learning approach using Dropout at test time as Monte Carlo sampling to model the pixel-wise uncertainty and to analyze the image information to improve the training performance. 

## Slides

![slides-fig]


* [[Slideshare slides]](https://es.slideshare.net/xavigiro/active-deep-learning-for-medical-imaging)

[slides-fig]: https://raw.githubusercontent.com/marc-gorriz/CEAL-Medical-Image-Segmentation/master/fig/slides.png "Project slides"

## Cost-Effective Active Learning methodology
A Cost-Effective Active Learning (CEAL) algorithm is able to interactively query the human annotator or the own ConvNet model (automatic annotations from high confidence predictions) new labeled instances from a pool of unlabeled data. Candidates to be labeled are chosen by estimating their uncertainty based on the stability of the pixel-wise predictions when a dropout is applied on a deep neural network. We trained the U-Net architecture using the CEAL methodology for solving the melanoma segmentation problem, obtaining pretty good results considering the lack of labeled data.
