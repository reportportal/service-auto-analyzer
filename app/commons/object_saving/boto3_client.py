#  Copyright 2023 EPAM Systems
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import Any, Generator

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.commons import logging
from app.commons.model.launch_objects import ApplicationConfig
from app.commons.object_saving.blob_storage import BlobStorage

LOGGER = logging.getLogger("analyzerApp.boto3Client")


class Boto3Client(BlobStorage):
    region: str
    s3_client: Any

    def __init__(self, app_config: ApplicationConfig) -> None:
        super().__init__(app_config)
        self.region = app_config.datastoreRegion or "us-east-1"

        # Build boto3 client configuration
        config_params = {
            "region_name": self.region,
        }

        # Only set credentials if explicitly provided, otherwise boto3 will use:
        # - IAM roles on EC2 instances
        # - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
        # - AWS credentials file (~/.aws/credentials)
        # - IAM roles for ECS tasks
        if app_config.datastoreAccessKey and app_config.datastoreSecretKey:
            config_params["aws_access_key_id"] = app_config.datastoreAccessKey
            config_params["aws_secret_access_key"] = app_config.datastoreSecretKey

        # Add endpoint_url if s3Endpoint is configured (for S3-compatible services or local testing)
        if app_config.datastoreEndpoint:
            config_params["endpoint_url"] = app_config.datastoreEndpoint

        self.s3_client = boto3.client("s3", **config_params)
        LOGGER.debug(f"Boto3 S3 client initialized with region {self.region}")

    def _paginate_objects(self, bucket_name: str, prefix: str) -> Generator[dict, None, None]:
        """Paginate through S3 objects with a given prefix.

        :param bucket_name: Name of the S3 bucket
        :param prefix: Prefix to filter objects
        :return: Generator yielding S3 object dictionaries from the paginated results
        """
        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            if "Contents" not in page:
                continue
            for obj in page["Contents"]:
                yield obj

    def remove_project_objects(self, bucket: str, object_names: list[str]) -> None:
        bucket_name = self.get_bucket(bucket)
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
        except ClientError:
            return

        for object_name in object_names:
            path = self.get_path(object_name, bucket_name, bucket)
            try:
                self.s3_client.delete_object(Bucket=bucket_name, Key=path)
            except ClientError as e:
                LOGGER.warning(f"Failed to delete object {path} from bucket {bucket_name}: {e}")

    def _create_bucket(self, bucket_name: str):
        LOGGER.debug(f"Creating S3 bucket {bucket_name}")
        try:
            bucket_config = {}
            if self.region != "us-east-1":
                # us-east-1 doesn't accept LocationConstraint
                bucket_config["CreateBucketConfiguration"] = {"LocationConstraint": self.region}
            self.s3_client.create_bucket(Bucket=bucket_name, **bucket_config)
            LOGGER.debug(f"Created S3 bucket {bucket_name}")
        except ClientError as e:
            if e.response["Error"]["Code"] != "BucketAlreadyOwnedByYou":
                raise

    def put_project_object(self, data: Any, bucket: str, object_name: str, using_json=False) -> None:
        bucket_name = self.get_bucket(bucket)
        path = self.get_path(object_name, bucket_name, bucket)

        # Create bucket if it doesn't exist
        if bucket_name:
            try:
                self.s3_client.head_bucket(Bucket=bucket_name)
            except ClientError as e:
                if e.response["Error"]["Code"] != "404":
                    raise e
                self._create_bucket(bucket_name)

        data_to_save, content_type = self.serialize_data(data, using_json)
        data_stream = self.create_data_stream(data_to_save)

        response = self.s3_client.put_object(
            Bucket=bucket_name,
            Key=path,
            Body=data_stream,
            ContentType=content_type,
        )
        etag = response.get("ETag", "")
        LOGGER.debug(f'Saved into bucket "{bucket_name}" with path "{path}", etag "{etag}": {data}')

    def get_project_object(self, bucket: str, object_name: str, using_json=False) -> object | None:
        bucket_name = self.get_bucket(bucket)
        path = self.get_path(object_name, bucket_name, bucket)
        try:
            response = self.s3_client.get_object(Bucket=bucket_name, Key=path)
            data = response["Body"].read()
        except (BotoCoreError, ClientError) as exc:
            raise ValueError(f'Unable to get file in bucket "{bucket_name}" with path "{path}"', exc)
        return self.deserialize_data(data, using_json)

    def does_object_exists(self, bucket: str, object_name: str) -> bool:
        bucket_name = self.get_bucket(bucket)
        path = self.get_path(object_name, bucket_name, bucket)

        if bucket_name:
            try:
                self.s3_client.head_bucket(Bucket=bucket_name)
            except ClientError:
                return False

        try:
            self.s3_client.head_object(Bucket=bucket_name, Key=path)
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise e
        return True

    def get_folder_objects(self, bucket: str, folder: str) -> list[str]:
        bucket_name = self.get_bucket(bucket)
        path = self.get_path(folder, bucket_name, bucket)

        if bucket_name:
            try:
                self.s3_client.head_bucket(Bucket=bucket_name)
            except ClientError:
                return []

        object_names = set()
        prefix = self.get_prefix(path)

        try:
            for obj in self._paginate_objects(bucket_name, prefix):
                object_name = self.extract_object_name(obj["Key"], folder, path)
                object_names.add(object_name)
        except ClientError as e:
            LOGGER.warning(f"Failed to list objects in bucket {bucket_name} with prefix {prefix}: {e}")
            return []

        return sorted(object_names)

    def remove_folder_objects(self, bucket: str, folder: str) -> bool:
        bucket_name = self.get_bucket(bucket)
        path = self.get_path(folder, bucket_name, bucket)

        if bucket_name:
            try:
                self.s3_client.head_bucket(Bucket=bucket_name)
            except ClientError:
                return False

        result = False
        prefix = self.get_prefix(path)

        try:
            for obj in self._paginate_objects(bucket_name, prefix):
                self.s3_client.delete_object(Bucket=bucket_name, Key=obj["Key"])
                result = True
        except ClientError as e:
            LOGGER.warning(f"Failed to remove folder objects in bucket {bucket_name} with prefix {prefix}: {e}")
            return False

        return result
