python2 distillation_main.py \
	--save-dir distillation_results \
	--teacher-checkpoint models/mosaic.pth \
	--learning-rate 0.05 \
	--coco \
	--coco-dataset ../coco2014
	#--style-image images/style_images/mosaic.jpg
	#--arch vgg11 \
	#--evaluate \
	#--slim-checkpoint /home/${user}/distill_results/checkpoint_86.tar

