JAN+DAN METHODS.

In JAN.ipynb file run the cell for the specific task that you want to test (OE or AR, balanced or unbalanced). for AR, the number of instances selected for balanced and unbalanced datasets are written in the comments.

Change the number of classes and input feature dimensions inside the code to fit the considered task.

TO TRAIN AND TEST DAN/JAN: !python train.py  --loss_name DAN_Linear --tradeoff 1.0 --using_bottleneck 1
YOU CAN CHANGE THE --loss_name PARAMETER TO CHOOSE THE WANTED METHOD.
