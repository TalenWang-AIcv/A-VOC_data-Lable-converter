#### An VOC_data Lable converter
This is a Lable converter. You can use it to convert VOC Data Lables from original VOC .txt. 

This code will find out the target imgs include the keywords that you have given.

And keep the .txt VOC lable '1', but convert VOC label '-1' to '0'.

Add a shell anno_coco2voc.sh, it can convert COCO dataset (*.json) files to VOC07 format (*.xml). Which references to https://github.com/CasiaFan/Dataset_to_VOC_converter, you can get the complete code there. 

Add a coco_data.py can convert (*.xml) to (*.txt) like VOC dataset. And you can use it to make data package in (*.pkl) format.
