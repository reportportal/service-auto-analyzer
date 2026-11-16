#  Copyright 2026 EPAM Systems
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Storage implementation based on Apache OpenDAL, compatible with ReportPortal service-api datastore settings."""

import os
from typing import Any

import opendal
from opendal import exceptions as opendal_exceptions

from app.commons import logging
from app.commons.model.launch_objects import ApplicationConfig
from app.commons.object_saving.blob_storage import PATH_SEPARATOR, BlobStorage
from app.commons.object_saving.storage import unify_path_separator
from app.utils.utils import put_if_set

LOGGER = logging.getLogger("analyzerApp.openDalClient")

S3_COMPATIBLE_TYPE = "s3-compatible"
AWS_S3_TYPE = "aws-s3"
AZURE_TYPE = "azure"
GCS_TYPE = "gcs"
DATASTORE_TYPES: tuple[str, ...] = (S3_COMPATIBLE_TYPE, AWS_S3_TYPE, AZURE_TYPE, GCS_TYPE)

# OpenDAL exceptions do not share a common base class except `Exception`
OPENDAL_ERRORS: tuple[type[Exception], ...] = (
    opendal_exceptions.Error,
    opendal_exceptions.Unexpected,
    opendal_exceptions.Unsupported,
    opendal_exceptions.ConfigInvalid,
    opendal_exceptions.NotFound,
    opendal_exceptions.PermissionDenied,
    opendal_exceptions.IsADirectory,
    opendal_exceptions.NotADirectory,
    opendal_exceptions.AlreadyExists,
    opendal_exceptions.IsSameFile,
    opendal_exceptions.ConditionNotMatch,
    opendal_exceptions.Conflict,
    opendal_exceptions.RangeNotSatisfied,
    opendal_exceptions.RateLimited,
)


def build_operator_config(app_config: ApplicationConfig) -> tuple[str, dict[str, str]]:
    """Build OpenDAL scheme and service options from the application configuration.

    The mapping follows ReportPortal service-api datastore configuration. User options from
    `DATASTORE_OPENDAL_OPTIONS` are applied last and override the mapped values.

    :param app_config: the application configuration
    :return: tuple of OpenDAL scheme and its options
    :raises ValueError: if the datastore type is not supported by OpenDAL client
    """
    datastore_type = app_config.datastoreType
    options: dict[str, str] = {}
    if datastore_type in (S3_COMPATIBLE_TYPE, AWS_S3_TYPE):
        scheme = "s3"
        options["bucket"] = app_config.datastoreDefaultBucketName
        put_if_set(options, "region", app_config.datastoreRegion)
        if datastore_type == S3_COMPATIBLE_TYPE:
            put_if_set(options, "endpoint", app_config.datastoreEndpoint)
        # Without credentials OpenDAL uses AWS default credential chain: environment variables, IAM roles, etc.
        if app_config.datastoreAccessKey and app_config.datastoreSecretKey:
            options["access_key_id"] = app_config.datastoreAccessKey
            options["secret_access_key"] = app_config.datastoreSecretKey
    elif datastore_type == AZURE_TYPE:
        scheme = "azblob"
        put_if_set(options, "container", app_config.datastoreAzureContainer)
        put_if_set(options, "account_name", app_config.datastoreAzureAccountName)
        put_if_set(options, "account_key", app_config.datastoreAzureAccountKey)
        put_if_set(options, "endpoint", app_config.datastoreAzureEndpoint)
    elif datastore_type == GCS_TYPE:
        scheme = "gcs"
        put_if_set(options, "bucket", app_config.datastoreGcsBucket)
        put_if_set(options, "credential_path", app_config.datastoreGcsCredentialsPath)
        put_if_set(options, "endpoint", app_config.datastoreGcsEndpoint)
    else:
        raise ValueError(
            f'Datastore type "{datastore_type}" is not supported by OpenDAL client, possible types are: '
            f"{DATASTORE_TYPES}"
        )
    options.update(app_config.datastoreOpendalOptions)
    return scheme, options


class OpenDalClient(BlobStorage):
    """Storage implementation based on Apache OpenDAL.

    One OpenDAL Operator is bound to one bucket (container). Buckets are not created by the client, they must exist.
    Project "buckets" which differ from the default bucket are stored as folders inside the bound bucket, the same way
    as service-api does: `prj-<id>/<path>`.
    """

    _default_bucket: str
    _operator: opendal.Operator

    def __init__(self, app_config: ApplicationConfig, operator: opendal.Operator | None = None) -> None:
        """Create OpenDAL client.

        :param app_config: the application configuration
        :param operator: OpenDAL Operator to use instead of the one built from the configuration
        """
        super().__init__(app_config)
        self._default_bucket = app_config.datastoreDefaultBucketName
        if operator is not None:
            self._operator = operator
        else:
            scheme, options = build_operator_config(app_config)
            self._operator = opendal.Operator(scheme, **options)
            LOGGER.debug(f'OpenDAL client initialized with scheme "{scheme}" for type "{app_config.datastoreType}"')

    def _get_key(self, bucket: str, object_name: str) -> str:
        """Get object key inside the bound bucket.

        :param bucket: project identifier
        :param object_name: name of the object
        :return: the object key
        """
        bucket_name = self.get_bucket(bucket)
        if bucket_name == self._default_bucket:
            return self.get_path(object_name, bucket_name, bucket)
        project_name = self._get_project_name(bucket)
        if not project_name:
            return unify_path_separator(object_name)
        return unify_path_separator(os.path.join(project_name, object_name))

    def remove_project_objects(self, bucket: str, object_names: list[str]) -> None:
        for object_name in object_names:
            key = self._get_key(bucket, object_name)
            try:
                self._operator.delete(key)
            except OPENDAL_ERRORS as exc:
                LOGGER.warning(f'Failed to delete object "{key}": {exc}')

    def put_project_object(self, data: Any, bucket: str, object_name: str, using_json: bool = False) -> None:
        key = self._get_key(bucket, object_name)
        data_to_save, content_type = self.serialize_data(data, using_json)
        self._operator.write(key, data_to_save, content_type=content_type)
        LOGGER.debug(f'Saved with path "{key}": {data}')

    def get_project_object(self, bucket: str, object_name: str, using_json: bool = False) -> object | None:
        key = self._get_key(bucket, object_name)
        try:
            data = self._operator.read(key)
        except OPENDAL_ERRORS as exc:
            raise ValueError(f'Unable to get file with path "{key}"', exc)
        return self.deserialize_data(bytes(data), using_json)

    def does_object_exists(self, bucket: str, object_name: str) -> bool:
        key = self._get_key(bucket, object_name)
        try:
            self._operator.stat(key)
        except opendal_exceptions.NotFound:
            return False
        return True

    def get_folder_objects(self, bucket: str, folder: str) -> list[str]:
        path = self._get_key(bucket, folder)
        prefix = self.get_prefix(path)
        object_names = set()
        try:
            for entry in self._operator.list(prefix):
                if entry.path == prefix:
                    continue
                object_names.add(self.extract_object_name(entry.path, folder, path))
        except OPENDAL_ERRORS as exc:
            LOGGER.warning(f'Failed to list objects with prefix "{prefix}": {exc}')
            return []
        return sorted(object_names)

    def remove_folder_objects(self, bucket: str, folder: str) -> bool:
        path = self._get_key(bucket, folder)
        prefix = self.get_prefix(path)
        result = False
        try:
            for entry in self._operator.list(prefix, recursive=True):
                if entry.path.endswith(PATH_SEPARATOR):
                    # Skip folders (and folder markers), only objects are removed
                    continue
                self._operator.delete(entry.path)
                result = True
        except OPENDAL_ERRORS as exc:
            LOGGER.warning(f'Failed to remove folder objects with prefix "{prefix}": {exc}')
            return False
        return result
