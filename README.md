<p align="center">
  <h1 align="center">GS2Mesh</h1>
  <p align="center"> </p>
  <h3 align="center"></h3>

</p>



### pre_trained stereo matching mdoel is on link(https://drive.google.com/drive/folders/1cZLcIjHlmUo986gkR6FbofG1cj5BT36x)
place 'defomstereo_vitl_sceneflow.pth' on'third_party/DEFOM-Stereo/checkpoints/defomstereo_vitl_sceneflow.pth'


### CUDA_VISIBLE_DEVICES=X python run_single.py --colmap_name example --renderer_baseline_percentage 15 --TSDF_min_depth_baselines 4 --TSDF_max_depth_baselines 50



### 결과가 이상하면 min_depth, max_depth 조정

