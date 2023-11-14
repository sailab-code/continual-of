# Continual Unsupervised Learning for Optical Flow Estimation with Deep Networks - COLLAs 2022

Authors: Simone Marullo, Matteo Tiezzi, Alessandro Betti, Lapo Faggi, Enrico Meloni, Stefano Melacci

[Paper link](https://proceedings.mlr.press/v199/marullo22a.html) 

_Notice that reproducibility is not guaranteed by PyTorch across different releases, platforms, hardware. Moreover,
determinism cannot be enforced due to use of PyTorch operations for which deterministic implementations do not exist
(e.g. bilinear upsampling)._

Make sure to have Python dependencies (except PyTorch) by running:
```
pip install -r requirements.txt
```

Concerning PyTorch, follow the [instructions](https://pytorch.org/get-started/previous-versions/#v171) on the official website.

REPOSITORY DESCRIPTION
----------------------

We provide the source code in a zip archive, once extracted the folder structure is the following:

    lve :                   main source folder
    run_colof.py :          experiments runner (continual online learning optical flow)
    run_baseline.py :       baseline runner (smurf, raft, hs, ..)
    reproduce_runs.txt :    text file containing the command lines (and parameters) to reproduce the main results

HOW TO GET PRERENDERED STREAMS
---------------------------------
Prerendered streams A, B, C are included in [this archive](https://drive.google.com/file/d/1SAviqsxp1pSB5WI_haO8GBx2dDlG7LCG/view?usp=sharing) on Google Drive.
If you have any issue with the download ('quota exceeded' message from Google Drive), please make sure that you are logged with your personal Google account.
You have to create in the current directory a new folder, called `data` and uncompress there the archive. Stream M is not publicly available.

The prerendered streams consists of two folders, `frames` and `motion`, containing frames and motion ground truth in subfolders of 100 steps. 
The `frames` folder contains PNG files, while `motion` contains binary gzipped numpy data arrays. 
If you would like to have additional details about the data format, please have a look at `stream_utils.py`: the class PairsOfFrameDataset is an example of external usage of these streams, being a PyTorch-like Dataset which returns tuples of the like (old_frame, new_frame, motion_ground_truth).

HOW TO REPRODUCE THE MAIN RESULTS
---------------------------------

_Concerning the baseline setup:_

- RAFT: Follow the instruction on the authors' [repository](https://github.com/princeton-vl/RAFT/). 
You need to clone the repository and consequently set the variables in `settings.py`. You can download the pretrained 'sintel' weights by using the downloader script they provide.
- SMURF: Follow the instruction on the authors' [repository](https://github.com/google-research/google-research/tree/master/smurf). 
You need to clone the repository and consequently set the variables in settings.py. You can download the pretrained 'sintel' weights by using gsutil as they describe.
- FlowNetS: You can download the flownets weights by using the downloader script `downloader_flownets.py` in this current repository.

After cloning the repositories, please make sure that the paths in `settings.py` are correct.

In the `reproduce_runs.txt` file there are the command lines (hence, the parameters) required to reproduce
the experiments of the main results (Table 1).

HOW TO RUN AN EXPERIMENT
------------------------
The script `run_colof.py` allows you to run an experiment on one of the visual streams presented in the paper,
which can be specified by the argument `--experience`.

The PyTorch device is chosen through the `--device` argument (`cpu`, `cuda:0`,
`cuda:1`, etc.).

Detailed arguments description:

      --step_size STEP_SIZE
                            learning rate of neural network optimizer
      --weight_decay WEIGHT_DECAY
                            weight decay of neural network optimizer
      --lambda_s LAMBDA_S   weight of the smoothness penalty in the loss function
      --charb_eps CHARB_EPS
                            epsilon parameter in charbonnier distance
      --charb_alpha CHARB_ALPHA
                            alpha parameter in charbonnier distance
      --recon_linf_thresh RECON_LINF_THRESH
                            threshold for reconstruction accuracy computation
                            (comma separated if you want to have multiple
                            thresholds
      --subsampling_updates SUBSAMPLING_UPDATES
                            update subsampling policies: integer [decimation
                            factor: 0 for no subsampling], avgflow(q, r,
                            warmup_frames, force_update_every_n_frames),
                            avgflowhistory(l, warmup_frames,
                            force_update_every_n_frames), avgdiff(q,
                            warmup_frames)
      --force_gray {yes,no}
                            convert stream data to grayscale
      --device DEVICE       computation device (cpu, cuda..)
      --iter_ihs ITER_IHS   HS iteration limit, only to be set for HS
      --warm_ihs WARM_IHS   HS warm start option, only to be set for HS
      --save {yes,no}       flag to create a separate folder for the current
                            experiment, 'yes' to persist the model
      --save_output {yes,no}
                            flag to save the flow predictions
      --freeze {yes,no}     flag to skip all the weights update operations
      --load LOAD           path for pre-loading models
      --exp_type {full,model-selection}
                            flag to indicate whether the current run is for model
                            selection (short portion of stream) or full experiment
      --training_loss {photo_and_smooth,hs}
                            neural network loss function (photometric+smoothness
                            or HS-like loss
      --port PORT           visualizer port
      --arch {none,none-ihs,resunetof,ndconvof,dilndconvof,flownets,sota-flownets,sota-smurf,sota-raft,sota-raft-small}
                            neural architecture for flow prediction
      --experience {a,b,c,movie,cat}
                            stream selection
      --output_folder OUTPUT_FOLDER
                            additional output folder
      --custom_playback CUSTOM_PLAYBACK
                            flag to alter the stream playback: 'x:y' means x
                            minutes of standard playback and y minutes interleaved
                            with y minutes of pause, repeated
      --verbose {yes,no}
      --seed SEED           seed for neural network weights initialization

The statistics of each experiment are visible in the visualizer port (look at the command output for a dynamically generated link) and printed on screen in the console.
They are also dumped in the `model_folder/results.json` file (you will find several JSON files corresponding to the different settings).

HOW TO RENDER TDW STREAMS
------------------------
Scenes of stream A, B and C are programmatically generated with scripts included in this archive. For extension or further customization purposes, they can be rendered with TDW.
You have to download the appropriate package for TDW build [here](https://github.com/threedworld-mit/tdw/releases/tag/v1.9.1). Then manually launch the build executable:
```
DISPLAY=:1 ./TDW.x86_64 -port=1071
```
and you can finally launch the scripts `tdw_stream_a.py`, `tdw_stream_b.py`, `tdw_stream_c.py` which are responsible for generating, controlling and storing the virtual world interactions.


Differences with the paper text
-------------------
Concerning the update subsampling policies, FlowMagnitude MAG is `avgflow`, Diff is `avgdiff`,  FlowDivergence DIV is `avgflowhistory`. See paper text for details.

Concerning the neural architectures, `none-ihs` is a dummy network with no weights that internally execute the iterative Horn-Schunck (HS) algorithm, 
while `sota-smurf`, `sota-raft`, `sota-flownets` are pretrained publicly available models (make sure to get the corresponding source code as previously described).
ResUnet, NdConv, DNdConv correspond to `resunetof`, `ndconvof`, `dilndconvof`. 

Concerning the metrics, Motion-F1 is termed `Moving-F1` here.


Acknowledgement
---------------

This software was developed in the context of some of the activities of the PRIN 2017 project RexLearn, funded by the Italian Ministry of Education, University and Research (grant no. 2017TWNMH2).This software was developed in the context of some of the activities of the PRIN 2017 project RexLearn, funded by the Italian Ministry of Education, University and Research (grant no. 2017TWNMH2).
