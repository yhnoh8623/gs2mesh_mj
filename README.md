<p align="center">
  <h1 align="center">GS2Mesh</h1>
  <p align="center"> </p>
  <h3 align="center"></h3>

</p>



### 사용법
data폴더에 example과 같이 데이터 업로드

3DGS 파일을 splatting_output/example/point_cloud/iteration_30000/point_cloud.ply에 업로드

CUDA_VISIBLE_DEVICES=X python run_single.py --colmap_name example --renderer_baseline_percentage 15 --TSDF_min_depth_baselines 4 --TSDF_max_depth_baselines 50 (--skip_masking)

masking 안할거면 --skip_masking 추가해서 코드 돌리기

결과가 이상하면 min_depth, max_depth, baseline_percentage 조정

