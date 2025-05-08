# components/model_pusher.py

from artifacts.artifact_entity import ModelTrainerArtifact, ModelPusherArtifact
import shutil
import os
import uuid

class ModelPusher:
    def __init__(self, trainer_artifact: ModelTrainerArtifact):
        self.trainer_artifact = trainer_artifact

    def initiate(self) -> ModelPusherArtifact:
        version_id = str(uuid.uuid4())
        saved_model_path = os.path.join("model_registry", f"{version_id}.pth")
        os.makedirs("model_registry", exist_ok=True)
        shutil.copy(self.trainer_artifact.model_path, saved_model_path)
        return ModelPusherArtifact(saved_model_path=saved_model_path, model_version=version_id)
