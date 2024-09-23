import pytorch_lightning as pl
import torch
import time
from gnn_tracking.preprocessing.point_cloud_builder import PointCloudBuilder
from gnn_tracking.utils.plotting import PointCloudPlotter
from gnn_tracking.utils.versioning import assert_version_geq


class GraphPipelineModule(pl.LightningModule):
    def __init__(self, data_indir):
        super(GraphPipelineModule, self).__init__()

    def preprocess_data(self, data):
        # Implement your data preprocessing here
        pc_builder = PointCloudBuilder(
        indir=data_path,
        outdir=outdir,
        n_sectors=1,
        pixel_only=True,
        redo=True,
        measurement_mode=False,
        sector_di=0,  # doesn't matter with n_sectors=1
        sector_ds=1.3,  # doesn't matter with n_sectors=1
        thld=0.9,  # Only matters for measurement mode
        log_level=0,
        detector_config=data_path / "detector_kaggle.csv",
        add_true_edges=True,
)
        # The stop argument limits us to process only 10 files
        pc_builder.process(stop=10)
        
        return processed_data

    def build_graph(self, data):
        # Implement your graph building here
        graph = data # Placeholder
        return graph

    def graph_classification(self, graph):
        # Implement your graph classification here
        classification_result = self.graph_classification_model(graph)
        return classification_result

    def postprocess(self, classification_result):
        # Implement your postprocessing here
        postprocessed_result = classification_result # Placeholder
        return postprocessed_result

    def forward(self, data):
        # This method brings all the steps together
        start_time = time.time()

        processed_data = self.preprocess_data(data)
        preprocess_time = time.time()

        graph = self.build_graph(processed_data)
        graph_time = time.time()

        classification_result = self.graph_classification(graph)
        classification_time = time.time()

        result = self.postprocess(classification_result)
        postprocess_time = time.time()

        total_time = postprocess_time - start_time
        print(f"Total Time: {total_time}s")
        print(f"Preprocessing Time: {preprocess_time - start_time}s")
        print(f"Graph Building Time: {graph_time - preprocess_time}s")
        print(f"Classification Time: {classification_time - graph_time}s")
        print(f"Postprocessing Time: {postprocess_time - classification_time}s")

        return result

    def training_step(self, batch, batch_idx):
        data, labels = batch
        outputs = self.forward(data)
        loss = self.compute_loss(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def compute_loss(self, outputs, labels):
        # Implement your loss computation here
        loss = torch.nn.functional.mse_loss(outputs, labels)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer