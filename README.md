BagOfKeyPoses
=============

Machine learning method based on a bag of key poses model and sequence alignment.



Description
-----------
This method handles multiclass classification of data with sequential or temporal relation. Any type of feature expressed as an array of double values can be used. This data has to be acquired in an ordered fashion, so that a sequence of features with a meaningful order can be obtained.

This method has successfully been employed for real-time recognition of human actions. In this case, spatial information is provided in the form of frame-based features that describe the human pose, whereas sequences of these features encode the temporal evolution. Therefore, this learning method applies spatio-temporal classification by means of learning the bag of key poses model and applying temporal alignment between the test sequence and the previously learned templates.



Details
-------
The method is explained in depth in _Chaaraoui, A. A., Climent-Pérez, P., & Flórez-Revuelta, F. (2013). Silhouette-based human action recognition using sequences of key poses. Pattern Recognition Letters, 34(15), 1799-1807. http://dx.doi.org/10.1016/j.patrec.2013.01.021_

Latest results achieved on RGB and RBGD-based human action recognition are published in _Chaaraoui, A. A., Padilla-López, J. R., Ferrández-Pastor, F. J., Nieto-Hidalgo, M., & Flórez-Revuelta, F. (2014). A Vision-Based System for Intelligent Monitoring: Human Behaviour Analysis and Privacy by Context. Sensors, 14(5), 8895-8925. http://dx.doi.org/10.3390/s140508895_



License
-------
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
-------

**_Alexandros Andre Chaaraoui_**  
Department of Computer Technology  
University of Alicante  
alexandros [AT] dtic [DOT] ua [DOT] es  
www.alexandrosandre.com  

**_Francisco Flórez-Revuelta_**  
Faculty of Science, Engineering and Computing  
Kingston University   
F [DOT] Florez [AT] kingston [DOT] ac [DOT] uk  
http://staffnet.kingston.ac.uk/~ku48824/

**_Pau Climent-Pérez_**  
Faculty of Science, Engineering and Computing  
Kingston University  
P [DOT] Climent [AT] kingston [DOT] ac [DOT] uk  



Results
-------

The following results have been achieved on publicly available datasets using this machine learning method for human action recognition. Please send an email to the authors if you want to add your results to this list.

Dataset | Article | LOSO | LOAO | LOVO | FPS
------------- | ------------- | ------------- | ------------- | ------------- | -------------
Weizmann | Chaaraoui, A. A., Padilla-López, J. R., Ferrández-Pastor, F. J., Nieto-Hidalgo, M., & Flórez-Revuelta, F. (2014) | 100%  | 100% | - | 197
MuHAVi-8 | Chaaraoui, A. A., Padilla-López, J. R., Ferrández-Pastor, F. J., Nieto-Hidalgo, M., & Flórez-Revuelta, F. (2014) | 100%  | 100% | 82.4% | 98
MuHAVi-14 | Chaaraoui, A. A., Padilla-López, J. R., Ferrández-Pastor, F. J., Nieto-Hidalgo, M., & Flórez-Revuelta, F. (2014) | 98.5%  | 94.1% | 59.6% | 99
MuHAVi-14 | Chaaraoui, A. A., & Flórez-Revuelta, F. (2014b) | 100%  | 100% | - | -
IXMAS | Chaaraoui, A. A., Padilla-López, J. R., Ferrández-Pastor, F. J., Nieto-Hidalgo, M., & Flórez-Revuelta, F. (2014) - | 91.4% | - | - | 207
DHA | Chaaraoui, A. A., Padilla-López, J. R., Ferrández-Pastor, F. J., Nieto-Hidalgo, M., & Flórez-Revuelta, F. (2014) | 95.2% | - | - | 99


LOSO = Leave-one-sequence-out cross validation  
LOAO = Leave-one-actor-out cross validation  
LOVO = Leave-one-view-out cross validation  



Related publications
--------------------

This method has also been used in the following publications. Please send an email to the authors if you want to add your publication to this list.

_Chaaraoui, A. A., & Flórez Revuelta, F. (2014a). Adaptive Human Action Recognition With an Evolving Bag of Key Poses. Autonomous Mental Development, IEEE Transactions on , vol.6, no.2, pp.139-152. http://dx.doi.org/10.1109/TAMD.2014.2315676_

_Chaaraoui, A. A. (2014). Vision-based Recognition of Human Behaviour for Intelligent Environments. PhD. Thesis. University of Alicante, Spain. http://hdl.handle.net/10045/36395_

_Chaaraoui, A. A., & Flórez-Revuelta, F. (2014b). Optimizing human action recognition based on a cooperative coevolutionary algorithm. Engineering Applications of Artificial Intelligence, 31, 116-125. http://dx.doi.org/10.1016/j.engappai.2013.10.003_

_Chaaraoui, A. A., Padilla-López, J. R., Climent-Pérez, P., & Flórez-Revuelta, F. (2014). Evolutionary joint selection to improve human action recognition with RGB-D devices. Expert Systems with Applications, 41(3), 786-794. http://dx.doi.org/10.1016/j.eswa.2013.08.009_

_Chaaraoui, A. A., Padilla-López, J. R., & Flórez-Revuelta, F. (2013). Fusion of Skeletal and Silhouette-based Features for Human Action Recognition with RGB-D Devices. In Computer Vision Workshops (ICCVW), 2013 IEEE International Conference on (pp. 91-97). IEEE. http://dx.doi.org/10.1109/ICCVW.2013.19_

_Chaaraoui, A. A., & Flórez-Revuelta, F. (2013). Human action recognition optimization based on evolutionary feature subset selection. In Proceeding of the fifteenth annual conference on Genetic and evolutionary computation conference (pp. 1229-1236). ACM. http://dx.doi.org/10.1145/2463372.2463529_

_Chaaraoui, A. A., Padilla-López, J. R., Ferrández-Pastor, F. J., García-Chamizo, J. M., Nieto-Hidalgo, M., Romacho-Agud, V., & Flórez-Revuelta, F. (2013). A Vision System for Intelligent Monitoring of Activities of Daily Living at Home. In Ambient Assisted Living and Active Aging (pp. 96-99). Springer International Publishing. http://dx.doi.org/10.1007/978-3-319-03092-0_14_

_Chaaraoui, A. A., Climent-Pérez, P., & Flórez-Revuelta, F. (2012). An efficient approach for multi-view human action recognition based on bag-of-key-poses. In Human Behavior Understanding (pp. 29-40). Springer Berlin Heidelberg. http://dx.doi.org/10.1007/978-3-642-34014-7_3_



Employed datasets
-----------------

**Weizmann**  
_Gorelick, L., Blank, M., Shechtman, E., Irani, M., & Basri, R. (2007). Actions as space-time shapes. Pattern Analysis and Machine Intelligence, IEEE Transactions on, 29(12), 2247-2253._

**MuHAVi-8 and MuHAVi-14**  
_Singh, S., Velastin, S. A., & Ragheb, H. (2010, August). Muhavi: A multicamera human action video dataset for the evaluation of action recognition methods. In Advanced Video and Signal Based Surveillance (AVSS), 2010 Seventh IEEE International Conference on (pp. 48-55). IEEE._

**IXMAS**  
_Weinland, D., Ronfard, R., & Boyer, E. (2006). Free viewpoint action recognition using motion history volumes. Computer Vision and Image Understanding, 104(2), 249-257._

**DHA**  
_Lin, Y. C., Hu, M. C., Cheng, W. H., Hsieh, Y. H., & Chen, H. M. (2012, October). Human action recognition and retrieval using sole depth information. In Proceedings of the 20th ACM international conference on Multimedia (pp. 1053-1056). ACM._
