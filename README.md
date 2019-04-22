# FlagsDetection

ln -s /userhome/apollo/Apollo_2/ /detectron/detectron/datasets/data/coco;

python /userhome/apollo/tools/train_net.py \
    --cfg /userhome/apollo/configs/apollo/add_city.yaml \
    OUTPUT_DIR /userhome/apollo/output_add_city | tee log.txt
