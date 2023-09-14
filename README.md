# HCR-OpenSource
the dataset and source code of paper "HCR-Auth: Reliable Bone Conduction Earphone Authentication with Head Contact Response"



mainDataset：

​	Data from 60 subjects(subject ID = folder name), each subject contributes 250 samples. 

​	Each sample has a length of 480, assembled by left channel transfer function (59 * 3), right channel transfer function (59 *3), Pearson's correlation coefficient(42 *3)



Source code: 

​	Run  "baseModelTraining.py"  to get the pre-trained base model "embedding_net.pth". The model is trained with data from subjects 46-60（15 subjects, regarded as default users）

​	Run "test_mainPerformance.py" to get the system performance under binaural channels, the samples number utilized to train user-specific model is set to 10.

​    More details could be seen within the source code.

​	

