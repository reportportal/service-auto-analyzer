"""
* Copyright 2019 EPAM Systems
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* https://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
"""

import json
import os
import sys
import uuid

import pika

LAUNCH_NAME = "Launch name"
ERROR_OCCURRED = "assertionError occurred"


# This file can be used for checking app functionality locally


class RpcClient():
    """RpcClient helps to use RPC type of communication with rabbitmq"""

    def __init__(self):
        self.connection = pika.BlockingConnection(
            pika.connection.
            URLParameters(os.getenv("AMQP_URL", "") + "?heartbeat=600"))
        self.response = None
        self.corr_id = None

        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=lambda channel, method, props, body: self._on_response(props, body),
            auto_ack=True)

    def _on_response(self, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, message, method):
        """RpcClient sends a message to a queue and waits for the answer"""
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='analyzer',
            routing_key=method,
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=message)

        while self.response is None:
            self.connection.process_data_events()
        return json.loads(self.response)

    def call_without_wait(self, message, method):
        """RpcClient sends a message to a queue and doesn't wait for the answer"""
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='analyzer',
            routing_key=method,
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=message)


rpc = RpcClient()

index_data = [{
    "launchId": 1,
    "project": 34,
    "launchName": "dsfdsf",
    "launchStartTime": [2021, 11, 8, 11, 23, 13],
    "analyzerConfig": {
        "minDocFreq": 1.0,
        "minTermFreq": 1.0,
        "minShouldMatch": 80,
        "numberOfLogLines": -1,
        "isAutoAnalyzerEnabled": True,
        "analyzerMode": "ALL",
        "indexingRunning": True,
    },
    "testItems": [{"testItemId": 2,
                   "uniqueId": "df",
                   "isAutoAnalyzed": False,
                   "issueType": "pb001",
                   "originalIssueType": "PB001",
                   "logs": [
                       {"logId": 3,
                        "logLevel": 40000,
                        "logTime": [2021, 11, 8, 11, 23, 13],
                        "message": ERROR_OCCURRED},
                       {"logId": 4,
                        "logLevel": 40000,
                        "logTime": [2021, 11, 9, 11, 23, 13],
                        "message": f"{ERROR_OCCURRED}\r\n error found\r\n error mined\r\n"}, ]
                   },
                  {"testItemId": 5,
                   "uniqueId": "df1",
                   "isAutoAnalyzed": False,
                   "issueType": "ti001",
                   "originalIssueType": "TI001",
                   "logs": [
                       {"logId": 5,
                        "logLevel": 40000,
                        "logTime": [2021, 11, 10, 11, 23, 13],
                        "message": f"{ERROR_OCCURRED}\r\n error found\r\n error mined"}, ]
                   }, ],
}, {
    "launchId": 2,
    "project": 34,
    "launchName": "dsfdsf",
    "launchStartTime": [2021, 9, 8, 11, 23, 13],
    "analyzerConfig": {
        "minDocFreq": 1.0,
        "minTermFreq": 1.0,
        "minShouldMatch": 80,
        "numberOfLogLines": -1,
        "isAutoAnalyzerEnabled": True,
        "analyzerMode": "ALL",
        "indexingRunning": True,
    },
    "testItems": [{"testItemId": 5,
                   "uniqueId": "df5",
                   "isAutoAnalyzed": False,
                   "issueType": "ab001",
                   "originalIssueType": "PB001",
                   "logs": [
                       {"logId": 6,
                        "logLevel": 40000,
                        "logTime": [2021, 9, 8, 11, 23, 13],
                        "message": ERROR_OCCURRED},
                       {"logId": 8,
                        "logLevel": 40000,
                        "logTime": [2021, 9, 9, 11, 23, 13],
                        "message": f"{ERROR_OCCURRED}\r\n error found\r\n error mined"}, ]
                   },
                  {"testItemId": 78,
                   "uniqueId": "df5",
                   "isAutoAnalyzed": False,
                   "issueType": "pb001",
                   "originalIssueType": "PB001",
                   "logs": [
                       {"logId": 45,
                        "logLevel": 40000,
                        "logTime": [2021, 9, 10, 11, 23, 13],
                        "message": ERROR_OCCURRED},
                       {"logId": 81,
                        "logLevel": 40000,
                        "logTime": [2021, 9, 11, 11, 23, 13],
                        "message": f"{ERROR_OCCURRED}\r\n error found\r\n error mined"}, ]
                   },
                  {"testItemId": 10,
                   "uniqueId": "df12",
                   "isAutoAnalyzed": False,
                   "issueType": "ab001",
                   "originalIssueType": "ab001",
                   "logs": [
                       {"logId": 38,
                        "logLevel": 40000,
                        "logTime": [2021, 9, 12, 11, 23, 13],
                        "message": f"{ERROR_OCCURRED}\r\n error found\r\n error mined"}]
                   },
                  {"testItemId": 15,
                   "uniqueId": "df",
                   "isAutoAnalyzed": False,
                   "issueType": "pb001",
                   "originalIssueType": "PB001",
                   "logs": [
                       {"logId": 555,
                        "logLevel": 40000,
                        "logTime": [2021, 9, 13, 11, 23, 13],
                        "message": ERROR_OCCURRED},
                       {"logId": 556,
                        "logLevel": 40000,
                        "logTime": [2021, 9, 14, 11, 23, 13],
                        "message": "nullpointerException occurred"}]
                   }],
}, {
    "launchId": 1,
    "project": 34,
    "launchName": "dsfdsf",
    "launchStartTime": [2021, 11, 8, 11, 23, 13],
    "analyzerConfig": {
        "minDocFreq": 1.0,
        "minTermFreq": 1.0,
        "minShouldMatch": 80,
        "numberOfLogLines": -1,
        "isAutoAnalyzerEnabled": True,
        "analyzerMode": "ALL",
        "indexingRunning": True,
    },
    "testItems": [{"testItemId": 23,
                   "uniqueId": "df",
                   "isAutoAnalyzed": False,
                   "issueType": "pb001",
                   "originalIssueType": "PB001",
                   "logs": [
                       {"logId": 32,
                        "logLevel": 40000,
                        "logTime": [2021, 11, 10, 11, 23, 13],
                        "message": ERROR_OCCURRED},
                       {"logId": 46,
                        "logLevel": 40000,
                        "logTime": [2021, 11, 11, 11, 23, 13],
                        "message": f"{ERROR_OCCURRED}\r\n error found\r\n error mined"}, ]
                   },
                  {"testItemId": 13,
                   "uniqueId": "df",
                   "isAutoAnalyzed": False,
                   "issueType": "pb001",
                   "originalIssueType": "PB001",
                   "logs": [
                       {"logId": 78,
                        "logLevel": 40000,
                        "logTime": [2021, 11, 12, 11, 23, 13],
                        "message": f"{ERROR_OCCURRED}\r\n error found\r\n error mined"},
                       {"logId": 113,
                        "logLevel": 40000,
                        "logTime": [2021, 11, 13, 12, 23, 13],
                        "message": "nullpointerException occurred \r\n error occurred \r\n error mined"}]
                   },
                  {"testItemId": 14,
                   "uniqueId": "df",
                   "isAutoAnalyzed": False,
                   "issueType": "pb001",
                   "originalIssueType": "PB001",
                   "logs": [
                       {"logId": 111,
                        "logLevel": 40000,
                        "logTime": [2021, 11, 14, 11, 23, 13],
                        "message": ERROR_OCCURRED},
                       {"logId": 112,
                        "logLevel": 40000,
                        "logTime": [2021, 11, 15, 11, 23, 13],
                        "message": "nullpointerException occurred"}]
                   }]}]

search_data = {
    "launchId": 4,
    "launchName": "dsfdsf",
    "itemId": 3,
    "projectId": 34,
    "filteredLaunchIds": [1],
    "logMessages": [f"{ERROR_OCCURRED} found mined", ],
    "logLines": -1, }

clean_index_data = {
    "ids": [3],
    "project": 34,
}

test_item_info = {
    "testItemId": 4,
    "uniqueId": "unique",
    "testCaseHash": 111,
    "launchId": 3,
    "launchName": LAUNCH_NAME,
    "project": 34,
    "analyzerConfig": {
        "minDocFreq": 1.0,
        "minTermFreq": 1.0,
        "minShouldMatch": 80,
        "numberOfLogLines": -1,
        "isAutoAnalyzerEnabled": True,
        "analyzerMode": "ALL",
        "indexingRunning": True,
    },
    "logs": [{"logId": 3,
              "logLevel": 40000,
              "message": ERROR_OCCURRED},
             {"logId": 4,
              "logLevel": 40000,
              "message": f"{ERROR_OCCURRED}\r\n error found\r\n error mined"}]
}

test_item_info_cluster = {
    "testItemId": 0,
    "uniqueId": "",
    "testCaseHash": 0,
    "launchId": 2,
    "launchName": LAUNCH_NAME,
    "project": 34,
    "analyzerConfig": {
        "minDocFreq": 1.0,
        "minTermFreq": 1.0,
        "minShouldMatch": 80,
        "numberOfLogLines": -1,
        "isAutoAnalyzerEnabled": True,
        "analyzerMode": "ALL",
        "indexingRunning": True,
    },
    "logs": [],
    "clusterId": 2202462168660536
}

remove_models_data = {
    "project": 34,
    "model_type": "auto_analysis",
}

defect_update_data = {
    "project": 34,
    "itemsToUpdate": {5: "pb001", 113: "ab001", 78: "si001"}
}

delete_test_items = {
    "project": 34,
    "itemsToDelete": [5, 78, 113]
}

delete_launches = {
    "project": 34,
    "launch_ids": [2, 42]
}

index_suggest_info_items = [{
    "project": 34,
    "testItem": 5,
    "testItemLogId": 1,
    "launchId": 2,
    "launchName": LAUNCH_NAME,
    "issueType": "pb001",
    "relevantItem": 3,
    "relevantLogId": 4,
    "isMergedLog": False,
    "matchScore": 80,
    "resultPosition": 1,
    "esScore": 1,
    "esPosition": 1,
    "modelFeatureNames": "",
    "modelFeatureValues": "",
    "modelInfo": "",
    "usedLogLines": -1,
    "minShouldMatch": 80,
    "userChoice": 1,
    "processedTime": 0.11,
    "methodName": "suggest",
    "clusterId": 2202462168660536
}]

remove_by_launch_start_time_data = {
    "project": 34,
    "interval_start_date": "2021-08-18 00:30:10",
    "interval_end_date": "2021-10-05 00:30:10",
}

remove_by_log_time_data = {
    "project": 34,
    "interval_start_date": "2021-09-11 00:30:10",
    "interval_end_date": "2021-11-09 00:30:10",
}

used_method = sys.argv[1] if len(sys.argv) > 1 else "index"
for_update = False
if len(sys.argv) > 2:
    for_update = True if sys.argv[2].lower() == "true" else False
number_lines = -1
if len(sys.argv) > 3:
    number_lines = int(sys.argv[3])
print(" [x] calling method %s" % used_method)
if used_method.strip() in ["delete"]:
    response = rpc.call("34", used_method)
elif used_method.strip() in ["clean"]:
    response = rpc.call(json.dumps(clean_index_data), used_method)
elif used_method.strip() in ["search"]:
    response = rpc.call(json.dumps(search_data), used_method)
elif used_method.strip() in ["suggest"]:
    response = rpc.call(json.dumps(test_item_info), used_method)
elif used_method.strip() in ["suggest_cluster"]:
    response = rpc.call(json.dumps(test_item_info_cluster), "suggest")
elif used_method.strip() in ["suggest_patterns"]:
    response = rpc.call("34", used_method)
elif used_method.strip() in ["remove_models"]:
    response = rpc.call(json.dumps(remove_models_data), used_method)
elif used_method.strip() in ["defect_update"]:
    response = rpc.call(json.dumps(defect_update_data), used_method)
elif used_method.strip() in ["item_remove"]:
    response = rpc.call(json.dumps(delete_test_items), used_method)
elif used_method.strip() in ["launch_remove"]:
    response = rpc.call(json.dumps(delete_launches), used_method)
elif used_method.strip() in ["index_suggest_info"]:
    response = rpc.call(json.dumps(index_suggest_info_items), used_method)
elif used_method.strip() in ["remove_by_launch_start_time"]:
    response = rpc.call(json.dumps(remove_by_launch_start_time_data), used_method)
elif used_method.strip() in ["remove_by_log_time"]:
    response = rpc.call(json.dumps(remove_by_log_time_data), used_method)
elif used_method.strip() in ["cluster"]:
    if not for_update:
        response = rpc.call(json.dumps({"launch": index_data[0],
                                        "project": 34,
                                        "forUpdate": for_update,
                                        "numberOfLogLines": number_lines}), used_method)
    else:
        response = rpc.call(json.dumps({"launch": index_data[1],
                                        "project": 34,
                                        "forUpdate": for_update,
                                        "numberOfLogLines": number_lines}), used_method)
else:
    response = rpc.call(json.dumps(index_data), used_method)
if used_method.strip() in ["index"]:
    rpc.call_without_wait(json.dumps(index_data), "namespace_finder")
    print("Namespace_finder info was processed")
print(" [.] Got %r" % response)
