
from pytorch_lightning import Trainer
import torch
from gnn_tracking.training.ml import MLModule
from gnn_tracking.training.tc import TCModule 
from gnn_tracking.training.ec import ECModule
from gnn_tracking.utils.loading import TrackingDataModule
from pytorch_lightning.profilers import PyTorchProfiler
from gnn_tracking.postprocessing.dbscanscanner import DBSCANHyperParamScanner
from gnn_tracking.metrics.losses.oc import CondensationLossTiger
from pathlib import Path

builder_params = {
    "outdir": "",
    "indir": "/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/codalab-data/part_1",
    "detector_config": "/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/codalab-data/detector_kaggle.csv",
    "n_sectors": 1,
    "redo": False,
    "pixel_only": False,
    "return_data": True,
    "write_output": False,
    "add_true_edges": False
    # Add other necessary params
}

class Timer:
    """
    :parameter:pipeline_step: The step in the pipeline
    :parameter:num_test_files: The number of files to test on
    :parameter:step_chkpt_path: The paths for the checkpoints of the trained model as a dictionary with the pipeline
    names as keys
    :parameter:step_data_path: The paths for the input data as a dictionary with the pipeline names as keys
    :parameter:study_name: A name to identify the study. This will be used to name the output files when you run the
    validation
    
    :returns: A torch trainer type that validation can be run on

          Ideally one would run a test instead of validation since validation has some extra checks, but the method
          hasn't been setup yet
         It shouldn't matter too much since the timer breaks down the individual steps

         After running the validation, the timer will write output to a folder in lightning_logs 
         It outputs a .json that can be opened with chrome or perfetto 
         Also outputs a .txt 

    Note: Putting the validation step in the class will fail because of a pre-hook error:
          RuntimeError: register_forward_pre_hook is not supported on ScriptModules
          Not sure why it works outside the class 
         
    """
    def __init__(self, pipeline_step, num_test_files, step_chkpt_path, step_data_path, study_name): 
        self.pipeline_step = pipeline_step
        self.num_test_files = num_test_files 
        
        self.chkpt_path = step_chkpt_path[pipeline_step]
        self.data_path = step_data_path[pipeline_step] 
        self.name = study_name
        
        self._check_paths() 
        self._check_gpu()
        self._get_module_name()
    
    def _check_paths(self): 
        assert self.chkpt_path.exists()
        assert self.data_path.exists()

    def _check_gpu(self): 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda": 
            print(torch.cuda.get_device_name(0)) 
        else: 
            print("couldn't find cuda, running on cpu") 

    def _get_module_name(self): 
        assert self.pipeline_step in ["gc", "oc", "ec", "full"]

        if self.pipeline_step == "gc": 
            self.module = MLModule
        elif self.pipeline_step == "oc" or self.pipeline_step == "full":
            self.module = TCModule

        else: 
            self.module = ECModule

    def _setup_data(self): 
        dm = TrackingDataModule(
            train=dict(
                dirs=[self.data_path],
                stop=1,
            ),
            val=dict(
                dirs=[self.data_path],
                start=1,
                stop=self.num_test_files,
                #cpus=24,
                num_workers=1,
                #batch_size=8,
                pin_memory=True
            ),
            predict=dict(
                dirs=[self.data_path],
                start=0,
                stop=self.num_test_files,
               #cpus=24,
                #num_workers=8,
                #batch_size=16,
                #pin_memory=True
            ),
            identifier="timing_study",
            #builder_params=builder_params,

            # could also configure a 'test' set here
            )
        return dm

    def setup_trainer(self): 
        # ckpt = torch.load(self.chkpt_path)
        # ckpt["state_dict"] = {
        #     key.replace("._orig_mod", ""): value
        #     for key, value in ckpt["state_dict"].items()
        # }
        #make_compatible(self.chkpt_path)
        lmodel = self.module.load_from_checkpoint(self.chkpt_path.parent / (self.chkpt_path.stem + self.chkpt_path.suffix), map_location=self.device)
        # really important to note, the default for active is three, meaning it'll only time three samples
        # If the number of active is too high, the postprocessing is extremely slow
        # 100 events will take > 25 min to process because tracing is slow. 10 should be fine
        custom_profiler = PyTorchProfiler(filename=self.name, export_to_chrome=True, profile_memory=True,
                                          schedule=torch.profiler.schedule(wait=1, warmup=10, active=1000, repeat=0))

        trainer = Trainer(accelerator=self.device, profiler=custom_profiler, inference_mode=True,
                          num_sanity_val_steps=0)
        dm = self._setup_data() 
        dm.setup(stage="predict")
        
        return dm, lmodel, trainer 

def make_compatible(path: Path) -> None:
    ckpt = torch.load(path, map_location="cpu")
    ckpt["state_dict"] = {
        key.replace("._orig_mod", ""): value
        for key, value in ckpt["state_dict"].items()
    }
    #c#kpt["state_dict"]["model._gtcn.hc_node_encoder._encoder.weight"] = ckpt["state_dict"].pop("model._gtcn.hc_node_encoder.layers.0.weight")
    #ckpt["state_dict"]["model._gtcn.hc_node_encoder._decoder.weight"] = ckpt["state_dict"].pop("model._gtcn.hc_node_encoder.layers.2.weight")

    new_path = path.parent / (path.stem + ".compat" + path.suffix)
    torch.save(ckpt, new_path)
