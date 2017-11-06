# Active Deep Learning for Medical Imaging Segmentation

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


[logo-vortex]: https://raw.githubusercontent.com/marc-gorriz/CEAL-Medical-Image-Segmentation/master/logos/Vortex.png "VORTEX Team (IRIT)"
[logo-enseeiht]: https://raw.githubusercontent.com/marc-gorriz/CEAL-Medical-Image-Segmentation/master/logos/enseeiht.png "Institut National polytechnique de Toulouse (ENSEEIHT)"
[logo-gpi]: https://raw.githubusercontent.com/marc-gorriz/CEAL-Medical-Image-Segmentation/master/logos/gpi.png "UPC GPI"

## Abstract

We propose a novel Active Learning framework capable to train effectively a convolutional neural network for semantic segmentation of medical imaging, with a limited amount of training labeled data. Our contribution is a practical Cost-Effective Active Learning approach using Dropout at test time as Monte Carlo sampling to model the pixel-wise uncertainty and to analyze the image information to improve the training performance. 

## Slides
<centrate>
<iframe src="//www.slideshare.net/slideshow/embed_code/key/cadu74MspLHLW5" width="595" height="485" frameborder="0" marginwidth="0" marginheight="0" scrolling="no" style="border:1px solid #CCC; border-width:1px; margin-bottom:5px; max-width: 100%;" allowfullscreen> </iframe> <div style="margin-bottom:5px"> <strong> <a href="//www.slideshare.net/xavigiro/active-deep-learning-for-medical-imaging" title="Active Deep Learning for Medical Imaging" target="_blank">Active Deep Learning for Medical Imaging</a> </strong> de <strong><a href="https://www.slideshare.net/xavigiro" target="_blank">Xavier Giro-i-Nieto</a></strong> </div>
</centrate>


## Cost-Effective Active Learning methodology
A Cost-Effective Active Learning (CEAL) algorithm is able to interactively query the human annotator or the own ConvNet model (automatic annotations from high confidence predictions) new labeled instances from a pool of unlabeled data. Candidates to be labeled are chosen by estimating their uncertainty based on the stability of the pixel-wise predictions when a dropout is applied on a deep neural network. We trained the U-Net architecture using the CEAL methodology for solving the melanoma segmentation problem, obtaining pretty good results considering the lack of labeled data.

![architecture-fig]

[architecture-fig]: https://raw.githubusercontent.com/marc-gorriz/CEAL-Medical-Image-Segmentation/master/logos/UncertainSamplingSelection.png

## Datasets
As explained in our work, all the tests were done with the [ISIC 2017 Challenge](https://challenge.kitware.com/#challenge/n/ISIC_2017%3A_Skin_Lesion_Analysis_Towards_Melanoma_Detection) dataset for Skin Lesion Analysis
towards melanoma detection, splitting the training set into labeled and unlabeled amount of data
to simulate the Active Learning problem with large amounts of unlabeled data at the beginning.

## Software frameworks: Keras
The model is implemented in [Keras](https://github.com/fchollet/keras/tree/master/keras), which at its time is developed over [TensorFlow](https://www.tensorflow.org). 

```
pip install -r https://github.com/marc-gorriz/CEAL-Medical-Image-Segmentation/blob/master/requeriments.txt
```


## Acknowledgements

We would like to especially thank Albert Gil Moreno from our technical support team at the Image Processing Group at the UPC.

| ![AlbertGil-photo]  |
|:-:|
| [Albert Gil](AlbertGil-web)   |

[AlbertGil-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/authors/AlbertGil.jpg "Albert Gil"
[JosepPujal-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/authors/JosepPujal.jpg "Josep Pujal"

[AlbertGil-web]: https://imatge.upc.edu/web/people/albert-gil-moreno
[JosepPujal-web]: https://imatge.upc.edu/web/people/josep-pujal

|   |   |
|:--|:-:|
|  We gratefully acknowledge the support of [NVIDIA Corporation](http://www.nvidia.com/content/global/global.php) with the donation of the GeoForce GTX [Titan X](http://www.geforce.com/hardware/desktop-gpus/geforce-gtx-titan-x) used in this work. |  ![logo-nvidia] |
|  The Image ProcessingGroup at the UPC is a [SGR14 Consolidated Research Group](https://imatge.upc.edu/web/projects/sgr14-image-and-video-processing-group) recognized and sponsored by the Catalan Government (Generalitat de Catalunya) through its [AGAUR](http://agaur.gencat.cat/en/inici/index.html) office. |  ![logo-catalonia] |

[logo-nvidia]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/nvidia.jpg "Logo of NVidia"
[logo-catalonia]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/generalitat.jpg "Logo of Catalan government"

## Contact

If you have any general doubt about our work or code which may be of interest for other researchers, please use the [public issues section](https://github.com/marc-gorriz/CEAL-Medical-Image-Segmentation/issues) on this github repo. Alternatively, drop us an e-mail at <mailto:xavier.giro@upc.edu>.
