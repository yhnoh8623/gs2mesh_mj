# =============================================================================
#  Imports
# =============================================================================

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

from gs2mesh_utils.argument_utils import ArgParser
from gs2mesh_utils.colmap_utils import extract_frames, create_downsampled_colmap_dir, run_colmap
from gs2mesh_utils.eval_utils import create_strings
from gs2mesh_utils.renderer_utils import Renderer
from gs2mesh_utils.stereo_utils import Stereo
from gs2mesh_utils.tsdf_utils import TSDF

device = 'cuda' if torch.cuda.is_available() else 'cpu'
base_dir = os.path.abspath(os.getcwd())

# =============================================================================
#  Run
# =============================================================================

def run_single(args):

    TSDF_voxel_length=args.TSDF_voxel/512
    colmap_dir = os.path.abspath(os.path.join(base_dir,'data',args.colmap_name))
    
    strings = create_strings(args)
    

    renderer = Renderer(base_dir,    # train/test 나누려면 여기서 eval=False
                        colmap_dir,
                        strings['output_dir_root'],
                        args,
                        dataset = strings['dataset'], 
                        splatting = strings['splatting'],
                        experiment_name = strings['experiment_name'],
                        device=device)

    # =============================================================================
    #  Prepare renderer
    # =============================================================================

    if not args.skip_rendering:
        renderer.prepare_renderer()

    # =============================================================================
    #  Initialize stereo
    # =============================================================================
    
    stereo = Stereo(base_dir, renderer, args, device=device)

    # =============================================================================
    #  Run stereo
    # =============================================================================
    
    if not args.skip_rendering:
        stereo.run(start=0, visualize=False, resolution = 1)


    if not args.skip_masking:
        args.TSDF_use_mask = True
        masks_dir = os.path.join(colmap_dir,'mask')
        masks_files = sorted([f for f in os.listdir(masks_dir) if os.path.isfile(os.path.join(masks_dir, f))], key=lambda x: x) 
        masks_files = [mask for mask in masks_files if mask[0]!='.']
        for i, mask_file in enumerate(tqdm(masks_files)):
            stereo_output_dir = renderer.render_folder_name(i)
            mask = plt.imread(os.path.join(masks_dir,mask_file))[:,:,0]
            plt.imsave(os.path.join(stereo_output_dir,'left_mask.png'), mask)
            np.save(os.path.join(stereo_output_dir,'left_mask.npy'), mask)


    # =============================================================================
    #  Initialize TSDF
    # =============================================================================
    
    tsdf = TSDF(renderer, stereo, args, args.colmap_name)

    if not args.skip_TSDF:
        # ================================================================================
        #  Run TSDF. the TSDF class will have an attribute "mesh" with the resulting mesh
        # ================================================================================
        
        tsdf.run(visualize = False, resolution = 1)

        # =============================================================================
        #  Save the original mesh before cleaning
        # =============================================================================
        
        tsdf.save_mesh()

        # =============================================================================
        #  Clean the mesh using clustering and save the cleaned mesh.
        # =============================================================================
        
        # original mesh is still available under tsdf.mesh (the cleaned is tsdf.clean_mesh)
        #tsdf.clean_mesh_for_sparse_view()
        tsdf.clean_mesh()

    # =============================================================================
    #  Return the path of the cleaned mesh for dataset evaluations
    # =============================================================================
    
    return os.path.join(renderer.output_dir_root, f'{tsdf.out_name}_cleaned_mesh.ply')

# =============================================================================
#  Main driver code with arguments
# =============================================================================

if __name__ == "__main__":
    parser = ArgParser('custom')
    args = parser.parse_args()
    #print(args)
    run_single(args)