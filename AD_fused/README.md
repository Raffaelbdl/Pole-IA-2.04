# NEAT + Transfuser = NEATFuser ?

This repository is the result of a school project on autonomous driving. As we aimed at understanding the state-of-the-art methods in use nowadays, we naturally ended up working on the CARLA environment, and studying its leaderboard.

Within this leaderboard, two methods studied different model architectures :
- [Transfuser](https://github.com/autonomousvision/transfuser)
- [NEAT](https://github.com/autonomousvision/neat)

As one is a powerful encoder for AD, and the other is an agile decoder, the idea to merge the projects quicly came around. That is the point of this repository;

### Dataset
To download the dataset, run
```
chmod +x download_data.sh
./download_data.sh
```
This is only a the part of the dataset used in (Transfuser)[https://github.com/autonomousvision/transfuser]. It contains the following informations:
```
- TownX_{tiny,short,long}: corresponding to different towns and routes files
    - routes_X: contains data for an individual route
        - rgb_{front, left, right, rear}: multi-view camera images at 400x300 resolution
        - seg_{front, left, right, rear}: corresponding segmentation images
        - depth_{front, left, right, rear}: corresponding depth images
        - lidar: 3d point cloud in .npy format
        - topdown: topdown segmentation images required for training LBC
        - 2d_bbs_{front, left, right, rear}: 2d bounding boxes for different agents in the corresponding camera view
        - 3d_bbs: 3d bounding boxes for different agents
        - affordances: different types of affordances
        - measurements: contains ego-agent's position, velocity and other metadata
```
For this implementation, only rgb_front, lidar, topdown and measurements are necessary, and a modification to the download bash file might be done in the future to have the rest deleted immediatly.

The full dataset is around 200 Go. Should be around 25 Go.

### Training
Run:
```
python train.py
```
### Validation
By default, validation is done after each epoch and saves the current model, and replaces the best if bested.
To launch a validation manually, run:
```
python eval.py
```
With visualization:
```
python eval.py --vis
```
This will store one input/output element per batch from the validation dataset. It's slow.


## Acknowledgments

Most of the code was taken and adapted from:
- [Transfuser](https://github.com/autonomousvision/transfuser)
- [NEAT](https://github.com/autonomousvision/neat)

All respect go to Aditya Prakash, Kashyap Chitta and Andreas Geiger.

Here are other several papers on autonomous driving from their group:
- [Chitta et al. - NEAT: Neural Attention Fields for End-to-End Autonomous Driving](https://arxiv.org/pdf/2109.04456.pdf)
- [Prakash et al. - Multi-Modal Fusion Transformer for End-to-End Autonomous Driving](https://arxiv.org/pdf/2104.09224.pdf)
- [Behl et al. - Label efficient visual abstractions for autonomous driving (IROS'20)](https://arxiv.org/pdf/2005.10091.pdf)
- [Ohn-Bar et al. - Learning Situational Driving (CVPR'20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ohn-Bar_Learning_Situational_Driving_CVPR_2020_paper.pdf)
- [Prakash et al. - Exploring Data Aggregation in Policy Learning for Vision-based Urban Autonomous Driving (CVPR'20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Prakash_Exploring_Data_Aggregation_in_Policy_Learning_for_Vision-Based_Urban_Autonomous_CVPR_2020_paper.pdf)