# TCRnet(beta)

Welcome, we want to introduce you TCRnet(beta version).

 TCRnet is a deep learning classifier based on convolutional neural network VGG-19. It can judge whether there is a disease based on the  subject's T cell receptor repertoires.
 
 
 TCRnet first performs t-sne dimensionality reduction on every tcr sequence. And then use the coding of the amino acid sequence of the CDR3 region as the third dimension of information. Then vgg-19 CNN network can extract the information in the vector precisely which can be used to classify
 
 ROC curve of TCRnet
 ![ROC](https://gitee.com/huhansan666666/picture/raw/master/img/20200612225711.png)

### 2020-05
TCRnet is now upgrading some core functions, so there may be instability, please leave a message to us, we will fix it as soon as possible
