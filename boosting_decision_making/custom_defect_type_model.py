from boosting_decision_making.defect_type_model import DefectTypeModel
from commons import minio_client


class CustomDefectTypeModel(DefectTypeModel):

    def __init__(self, app_config, project_id, folder=""):
        self.project_id = project_id
        self.minio_client = minio_client.MinioClient(app_config)
        super(CustomDefectTypeModel, self).__init__(folder=folder)
        self.is_global = False

    def load_model(self, folder):
        self.count_vectorizer_models = self.minio_client.get_project_object(
            self.project_id, folder + "count_vectorizer_models", using_json=False)
        self.models = self.minio_client.get_project_object(
            self.project_id, folder + "models", using_json=False)

    def save_model(self, folder):
        self.minio_client.put_project_object(
            self.count_vectorizer_models,
            self.project_id, folder + "count_vectorizer_models", using_json=False)
        self.minio_client.put_project_object(
            self.models,
            self.project_id, folder + "models", using_json=False)
