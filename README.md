BagOfKeyPoses
=============

Machine learning method based on a bag of key poses model and temporal alignment.


Description
===========
This method handles multiclass classification of data with sequential or temporal relation. Any type of feature expressed as an array of double values can be used. This data has to be acquired in an ordered fashion, so that a sequence of features with a meaningful order can be obtained.

This method has successfully been employed for real-time recognition of human actions. In this case, spatial information is provided in the form of frame-based features that describe the human pose, whereas sequences of these features encode the temporal evolution. Therefore, this learning method applies spatio-temporal classification by means of learning the bag of key poses model and applying temporal alignment between the test sequence and the previously learned templates.

Details
=======
The method is explained in depth in _Chaaraoui, A. A., Climent-Pérez, P., & Flórez-Revuelta, F. (2013). Silhouette-based human action recognition using sequences of key poses. Pattern Recognition Letters, 34(15), 1799-1807. http://dx.doi.org/10.1016/j.patrec.2013.01.021_

Latest results achieved on RGB and RBGD-based human action recognition are published in _Chaaraoui, A. A., Padilla-López, J. R., Ferrández-Pastor, F. J., Nieto-Hidalgo, M., & Flórez-Revuelta, F. (2014). A Vision-Based System for Intelligent Monitoring: Human Behaviour Analysis and Privacy by Context. Sensors, 14(5), 8895-8925. http://dx.doi.org/10.3390/s140508895_

License
=======
Distributed under the free software license Apache 2.0 License http://www.apache.org/licenses/LICENSE-2.0.html requiring  preservation of the copyright notice and disclaimer.

**If used in research work a citation to the following bibtex is required:**
```
  @article{Chaaraoui2013,
    title = "Silhouette-based human action recognition using sequences of key poses ",
    journal = "Pattern Recognition Letters ",
    volume = "34",
    number = "15",
    pages = "1799 - 1807",
    year = "2013",
    note = "Smart Approaches for Human Action Recognition ",
    issn = "0167-8655",
    doi = "http://dx.doi.org/10.1016/j.patrec.2013.01.021",
    url = "http://www.sciencedirect.com/science/article/pii/S0167865513000342",
    author = "Alexandros Andre Chaaraoui and Pau Climent-Pérez and Francisco Flórez-Revuelta",
  }
```

Authors
=======

*_Alexandros Andre Chaaraoui_*  
Department of Computer Technology  
University of Alicante  
alexandros [AT] dtic [DOT] ua [DOT] es  
www.alexandrosandre.com  

*_Francisco Flórez-Revuelta_*  
Faculty of Science, Engineering and Computing  
Kingston University   
F [DOT] Florez [AT] kingston [DOT] ac [DOT] uk  
http://www.dtic.ua.es/~florez/  

*_Pau Climent-Pérez_*  
Faculty of Science, Engineering and Computing  
Kingston University  
P [DOT] Climent [AT] kingston [DOT] ac [DOT] uk  
