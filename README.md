# FlagsDetection


trap in process:

flags2coco: In the beginning , forgot to create "ann['segmentation']", which didn't affect training process but necessary for initial training.

training: Problems have not solved till now, training log is weired, which was always being 0.75 in the beginning, broking after thousands of iters.
sometimes training accuracy ended with 0.99 but performed bad in test pictures when using res101-fpn, 
when using res50-c4, training log was right but performed bad in test pictures either.

docker build: I couldn't build the docker with original Dockerfile. Afer reading issus in github, I did some change in Dockerfile and it worked.

run failed: Changed the version of pyyaml in requirements.txt, from pyyaml>=3.12 to pyyaml==3.12

